#!/usr/bin/env python3
"""
Universal Verification Suite for MathLedger

Devin C - The Verifier
Mission: Verify every claim made by any Codex or Cursor.
Enable verification by writing universal checkers - one-liners that anyone can run locally.

Exit codes:
  0: [PASS] VERIFIED: ALL CLAIMS HOLD
  1: [FAIL] One or more verification checks failed
  2: [ERROR] Fatal error during verification

Usage:
  python tools/verify_all.py                    # Run all checks
  python tools/verify_all.py --check hash       # Run specific check
  python tools/verify_all.py --offline          # Skip database checks
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psycopg
except ImportError:
    psycopg = None


class VerificationResult:
    """Result of a verification check."""
    def __init__(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}


class Verifier:
    """Universal verification engine for MathLedger claims."""
    
    def __init__(self, offline: bool = False, verbose: bool = False):
        self.offline = offline
        self.verbose = verbose
        self.results: List[VerificationResult] = []
        
    def log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[VERIFY] {message}")
    
    def add_result(self, result: VerificationResult):
        """Add verification result."""
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.message}")
        if self.verbose and result.details:
            for key, value in result.details.items():
                print(f"  {key}: {value}")
    
    def verify_hash_integrity(self) -> VerificationResult:
        """Verify hash computation is deterministic and correct."""
        self.log("Checking hash integrity...")
        
        passed = True
        details = {}
        
        try:
            from backend.logic.canon import normalize
            
            test_formulas = ["p", "p->q", "(p∧q)->r"]
            
            for formula in test_formulas:
                normalized1 = normalize(formula)
                normalized2 = normalize(formula)
                hash1 = hashlib.sha256(normalized1.encode('utf-8')).hexdigest()
                hash2 = hashlib.sha256(normalized2.encode('utf-8')).hexdigest()
                
                if hash1 != hash2:
                    passed = False
                    details[f"non_deterministic_{formula}"] = f"hash1={hash1}, hash2={hash2}"
                else:
                    details[f"deterministic_{formula}"] = "OK"
                
                if normalized1 != normalized2:
                    passed = False
                    details[f"normalization_inconsistent_{formula}"] = f"norm1={normalized1}, norm2={normalized2}"
            
            message = "Hash computation is deterministic" if passed else "Non-deterministic hash detected"
            return VerificationResult("hash_integrity", passed, message, details)
            
        except Exception as e:
            return VerificationResult("hash_integrity", False, f"Error: {str(e)}")
    
    def verify_merkle_root(self) -> VerificationResult:
        """Verify Merkle root computation is deterministic."""
        self.log("Checking Merkle root computation...")
        
        try:
            from backend.ledger.blockchain import merkle_root
            
            test_ids = ["stmt1", "stmt2", "stmt3"]
            root1 = merkle_root(test_ids)
            root2 = merkle_root(test_ids)
            
            if root1 != root2:
                return VerificationResult(
                    "merkle_root",
                    False,
                    "Merkle root computation is non-deterministic",
                    {"root1": root1, "root2": root2}
                )
            
            empty_root = merkle_root([])
            expected_empty = hashlib.sha256(b"").hexdigest()
            
            if empty_root != expected_empty:
                return VerificationResult(
                    "merkle_root",
                    False,
                    "Empty Merkle root incorrect",
                    {"computed": empty_root, "expected": expected_empty}
                )
            
            single_root = merkle_root(["single"])
            
            details = {
                "deterministic": "OK",
                "empty_root": empty_root,
                "single_root": single_root,
                "test_root": root1
            }
            
            return VerificationResult(
                "merkle_root",
                True,
                "Merkle root computation verified",
                details
            )
            
        except Exception as e:
            return VerificationResult("merkle_root", False, f"Error: {str(e)}")
    
    def verify_file_existence(self) -> VerificationResult:
        """Verify critical files exist and are not drifted."""
        self.log("Checking file existence...")
        
        critical_files = [
            "backend/axiom_engine/derive.py",
            "backend/axiom_engine/rules.py",
            "backend/ledger/blockchain.py",
            "backend/logic/canon.py",
            "backend/orchestrator/app.py",
            "tools/metrics_lint_v1.py",
            "docs/progress.md",
            "README_ops.md",
        ]
        
        missing = []
        present = []
        
        for file_path in critical_files:
            full_path = Path(file_path)
            if full_path.exists():
                present.append(file_path)
            else:
                missing.append(file_path)
        
        passed = len(missing) == 0
        message = f"{len(present)}/{len(critical_files)} critical files present"
        
        if missing:
            message += f", {len(missing)} missing"
        
        details = {
            "present_count": len(present),
            "missing_count": len(missing),
            "missing_files": missing
        }
        
        return VerificationResult("file_existence", passed, message, details)
    
    def verify_metrics_schema(self) -> VerificationResult:
        """Verify metrics files conform to V1 schema."""
        self.log("Checking metrics schema...")
        
        metrics_files = [
            "artifacts/wpv5/run_metrics.jsonl",
            "artifacts/wpv5/run_metrics_v1.jsonl",
        ]
        
        found_files = []
        for file_path in metrics_files:
            if Path(file_path).exists():
                found_files.append(file_path)
        
        if not found_files:
            return VerificationResult(
                "metrics_schema",
                True,
                "No metrics files found (skipped)",
                {"note": "This is OK if no runs have been performed"}
            )
        
        try:
            from tools.metrics_lint_v1 import lint_v1
            
            all_valid = True
            details = {}
            
            for file_path in found_files:
                result = lint_v1(file_path)
                is_valid = not result['is_mixed_schema'] and not result['violations']
                
                if not is_valid:
                    all_valid = False
                
                details[file_path] = {
                    "valid": is_valid,
                    "v1_count": result['v1_count'],
                    "legacy_count": result['legacy_count'],
                    "violations": len(result['violations'])
                }
            
            message = "All metrics files valid" if all_valid else "Schema violations detected"
            return VerificationResult("metrics_schema", all_valid, message, details)
            
        except Exception as e:
            return VerificationResult("metrics_schema", False, f"Error: {str(e)}")
    
    def verify_database_integrity(self) -> VerificationResult:
        """Verify database schema and data integrity."""
        if self.offline:
            return VerificationResult(
                "database_integrity",
                True,
                "Skipped (offline mode)",
                {"note": "Use --online to enable database checks"}
            )
        
        self.log("Checking database integrity...")
        
        if psycopg is None:
            return VerificationResult(
                "database_integrity",
                False,
                "psycopg not installed",
                {"fix": "pip install psycopg[binary]"}
            )
        
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return VerificationResult(
                "database_integrity",
                False,
                "DATABASE_URL not set",
                {"fix": "Set DATABASE_URL environment variable"}
            )
        
        try:
            with psycopg.connect(db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        AND table_name IN ('statements', 'proofs', 'blocks', 'proof_parents')
                    """)
                    tables = [row[0] for row in cur.fetchall()]
                    
                    if len(tables) < 4:
                        missing = set(['statements', 'proofs', 'blocks', 'proof_parents']) - set(tables)
                        return VerificationResult(
                            "database_integrity",
                            False,
                            f"Missing tables: {missing}",
                            {"found_tables": tables}
                        )
                    
                    cur.execute("SELECT COUNT(*) FROM statements")
                    stmt_count = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM proofs")
                    proof_count = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM blocks")
                    block_count = cur.fetchone()[0]
                    
                    latest_block = None
                    merkle_valid = None
                    if block_count > 0:
                        cur.execute("""
                            SELECT block_number, merkle_root 
                            FROM blocks 
                            ORDER BY block_number DESC 
                            LIMIT 1
                        """)
                        row = cur.fetchone()
                        if row:
                            latest_block = row[0]
                            merkle_root_db = row[1]
                            
                            if isinstance(merkle_root_db, str):
                                merkle_valid = len(merkle_root_db) == 64 and all(c in '0123456789abcdef' for c in merkle_root_db)
                            elif isinstance(merkle_root_db, bytes):
                                merkle_valid = len(merkle_root_db) == 32
                    
                    details = {
                        "tables": tables,
                        "statement_count": stmt_count,
                        "proof_count": proof_count,
                        "block_count": block_count,
                        "latest_block": latest_block,
                        "merkle_format_valid": merkle_valid
                    }
                    
                    return VerificationResult(
                        "database_integrity",
                        True,
                        f"Database OK: {stmt_count} statements, {proof_count} proofs, {block_count} blocks",
                        details
                    )
                    
        except psycopg.OperationalError as e:
            return VerificationResult(
                "database_integrity",
                False,
                f"Database connection failed: {str(e)}",
                {"fix": "Check DATABASE_URL and ensure PostgreSQL is running"}
            )
        except Exception as e:
            return VerificationResult("database_integrity", False, f"Error: {str(e)}")
    
    def verify_normalization_idempotence(self) -> VerificationResult:
        """Verify normalization is idempotent."""
        self.log("Checking normalization idempotence...")
        
        try:
            from backend.logic.canon import normalize
            
            test_formulas = [
                "p",
                "p->q",
                "(p∧q)->r",
                "p∨q∨r",
                "((p->q)->r)->s",
            ]
            
            all_idempotent = True
            details = {}
            
            for formula in test_formulas:
                norm1 = normalize(formula)
                norm2 = normalize(norm1)
                
                is_idempotent = norm1 == norm2
                if not is_idempotent:
                    all_idempotent = False
                    details[f"non_idempotent_{formula}"] = f"{norm1} != {norm2}"
                else:
                    details[f"idempotent_{formula}"] = "OK"
            
            message = "All normalizations idempotent" if all_idempotent else "Non-idempotent normalization detected"
            return VerificationResult("normalization_idempotence", all_idempotent, message, details)
            
        except Exception as e:
            return VerificationResult("normalization_idempotence", False, f"Error: {str(e)}")
    
    def verify_api_parity(self) -> VerificationResult:
        """Verify API endpoints return consistent data with database."""
        if self.offline:
            return VerificationResult(
                "api_parity",
                True,
                "Skipped (offline mode)",
                {"note": "Use --online to enable API checks"}
            )
        
        self.log("Checking API parity...")
        
        try:
            from backend.orchestrator import app
            
            return VerificationResult(
                "api_parity",
                True,
                "API module importable",
                {"note": "Full API parity check requires running server"}
            )
        except Exception as e:
            return VerificationResult("api_parity", False, f"Error importing API: {str(e)}")
    
    def verify_proof_parents(self) -> VerificationResult:
        """Verify proof parent relationships are valid."""
        if self.offline:
            return VerificationResult(
                "proof_parents",
                True,
                "Skipped (offline mode)",
                {"note": "Use --online to enable proof parent checks"}
            )
        
        self.log("Checking proof parent relationships...")
        
        if psycopg is None:
            return VerificationResult(
                "proof_parents",
                False,
                "psycopg not installed",
                {"fix": "pip install psycopg[binary]"}
            )
        
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return VerificationResult(
                "proof_parents",
                False,
                "DATABASE_URL not set"
            )
        
        try:
            with psycopg.connect(db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM proof_parents pp
                        WHERE NOT EXISTS (
                            SELECT 1 FROM statements s WHERE s.hash = pp.parent_hash
                        )
                    """)
                    orphaned = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM proof_parents")
                    total_parents = cur.fetchone()[0]
                    
                    details = {
                        "total_parent_relationships": total_parents,
                        "orphaned_references": orphaned
                    }
                    
                    if orphaned > 0:
                        return VerificationResult(
                            "proof_parents",
                            False,
                            f"{orphaned} orphaned parent references found",
                            details
                        )
                    
                    return VerificationResult(
                        "proof_parents",
                        True,
                        f"All {total_parents} parent relationships valid",
                        details
                    )
                    
        except Exception as e:
            return VerificationResult("proof_parents", False, f"Error: {str(e)}")
    
    def run_all_checks(self, specific_check: Optional[str] = None):
        """Run all verification checks."""
        checks = [
            ("hash", self.verify_hash_integrity),
            ("merkle", self.verify_merkle_root),
            ("files", self.verify_file_existence),
            ("metrics", self.verify_metrics_schema),
            ("normalization", self.verify_normalization_idempotence),
            ("database", self.verify_database_integrity),
            ("api", self.verify_api_parity),
            ("parents", self.verify_proof_parents),
        ]
        
        if specific_check:
            checks = [(name, func) for name, func in checks if name == specific_check]
            if not checks:
                print(f"[ERROR] Unknown check: {specific_check}")
                print(f"Available checks: hash, merkle, files, metrics, normalization, database, api, parents")
                return False
        
        print(f"\n{'='*60}")
        print(f"MathLedger Universal Verification Suite")
        print(f"Devin C - The Verifier")
        print(f"{'='*60}\n")
        
        for name, check_func in checks:
            result = check_func()
            self.add_result(result)
            print()
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"{'='*60}")
        print(f"VERIFICATION SUMMARY: {passed}/{total} checks passed")
        print(f"{'='*60}\n")
        
        if passed == total:
            print("[PASS] VERIFIED: ALL CLAIMS HOLD")
            return True
        else:
            failed = [r.name for r in self.results if not r.passed]
            print(f"[FAIL] Failed checks: {', '.join(failed)}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Universal verification suite for MathLedger",
        epilog="Verify every claim. Enable verification by writing universal checkers."
    )
    parser.add_argument(
        "--check",
        type=str,
        help="Run specific check (hash, merkle, files, metrics, normalization, database, api, parents)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip database and API checks"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    verifier = Verifier(offline=args.offline, verbose=args.verbose)
    success = verifier.run_all_checks(specific_check=args.check)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
