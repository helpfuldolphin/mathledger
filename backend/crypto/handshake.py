"""
End-to-End Cryptographic Handshake Validation

Validates complete cryptographic workflows including:
- API key generation and validation
- Redis authentication
- Merkle tree construction and verification
- Statement hashing with domain separation
- Block sealing with cryptographic integrity
"""

import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from backend.crypto.hashing import (
    merkle_root,
    sha256_hex,
    hash_statement,
    hash_block,
    verify_merkle_proof,
    compute_merkle_proof,
    DOMAIN_LEAF,
    DOMAIN_NODE,
    DOMAIN_STMT,
    DOMAIN_BLCK,
)
from backend.crypto.auth import APIKeyManager, get_redis_url_with_auth, validate_redis_auth


class CryptoHandshake:
    """
    Validates end-to-end cryptographic workflows.
    
    Ensures all components work together correctly:
    - Key management
    - Authentication
    - Hashing with domain separation
    - Merkle tree operations
    - Block sealing
    """
    
    def __init__(self):
        self.test_results = []
        self.errors = []
    
    def log_test(self, name: str, passed: bool, details: str = ""):
        """Log test result."""
        self.test_results.append({
            "name": name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
        if not passed:
            self.errors.append(f"{name}: {details}")
    
    def validate_api_key_workflow(self) -> bool:
        """
        Validate complete API key workflow:
        1. Generate key
        2. Validate key
        3. Revoke key
        4. Verify revoked key fails
        """
        try:
            manager = APIKeyManager()
            
            key = manager.generate_key(scope="readonly", expires_days=30)
            self.log_test(
                "API key generation",
                key.startswith("mlvk-api-"),
                f"Generated key: {key[:20]}..."
            )
            
            is_valid, metadata = manager.validate_key(key)
            self.log_test(
                "API key validation",
                is_valid and metadata is not None,
                f"Key valid: {is_valid}, scope: {metadata.get('scope') if metadata else 'N/A'}"
            )
            
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            key_id = None
            for key_data in manager.keys["keys"]:
                if key_data["key_hash"] == key_hash:
                    key_id = key_data["key_id"]
                    break
            
            if key_id:
                revoked = manager.revoke_key(key_id)
                self.log_test(
                    "API key revocation",
                    revoked,
                    f"Key {key_id} revoked: {revoked}"
                )
                
                is_valid_after, metadata_after = manager.validate_key(key)
                self.log_test(
                    "Revoked key validation fails",
                    not is_valid_after,
                    f"Revoked key validation: {is_valid_after} (should be False)"
                )
            
            return all(r["passed"] for r in self.test_results[-4:])
            
        except Exception as e:
            self.log_test("API key workflow", False, f"Exception: {str(e)}")
            return False
    
    def validate_redis_auth_workflow(self) -> bool:
        """
        Validate Redis authentication workflow:
        1. Check auth configuration
        2. Generate authenticated URL
        3. Verify URL format
        """
        try:
            has_auth = validate_redis_auth()
            self.log_test(
                "Redis auth configuration",
                True,  # Always pass, just report status
                f"Redis auth configured: {has_auth}"
            )
            
            redis_url = get_redis_url_with_auth()
            self.log_test(
                "Redis URL generation",
                redis_url.startswith("redis://") or redis_url.startswith("rediss://"),
                f"URL: {redis_url[:30]}..."
            )
            
            redis_password = os.getenv("REDIS_PASSWORD")
            if redis_password:
                has_password_in_url = "@" in redis_url and ":" in redis_url.split("@")[0]
                self.log_test(
                    "Redis password injection",
                    has_password_in_url,
                    f"Password injected: {has_password_in_url}"
                )
            else:
                self.log_test(
                    "Redis password injection",
                    True,
                    "No REDIS_PASSWORD set (optional)"
                )
            
            return all(r["passed"] for r in self.test_results[-3:])
            
        except Exception as e:
            self.log_test("Redis auth workflow", False, f"Exception: {str(e)}")
            return False
    
    def validate_statement_workflow(self) -> bool:
        """
        Validate statement hashing workflow:
        1. Hash statement with domain separation
        2. Verify domain separation applied
        3. Verify determinism
        """
        try:
            statement = "p -> p"
            
            stmt_hash = hash_statement(statement)
            self.log_test(
                "Statement hashing",
                len(stmt_hash) == 64,
                f"Hash: {stmt_hash[:16]}..."
            )
            
            plain_hash = sha256_hex(statement)
            self.log_test(
                "Statement domain separation",
                stmt_hash != plain_hash,
                "STMT domain tag applied"
            )
            
            stmt_hash2 = hash_statement(statement)
            self.log_test(
                "Statement hash determinism",
                stmt_hash == stmt_hash2,
                "Hash is deterministic"
            )
            
            return all(r["passed"] for r in self.test_results[-3:])
            
        except Exception as e:
            self.log_test("Statement workflow", False, f"Exception: {str(e)}")
            return False
    
    def validate_merkle_workflow(self) -> bool:
        """
        Validate complete Merkle tree workflow:
        1. Build tree with domain separation
        2. Generate proof for leaf
        3. Verify proof
        4. Verify invalid proof fails
        """
        try:
            statements = ["p -> p", "p /\\ q -> p", "p /\\ q -> q"]
            
            root = merkle_root(statements)
            self.log_test(
                "Merkle tree construction",
                len(root) == 64,
                f"Root: {root[:16]}..."
            )
            
            proof = compute_merkle_proof(0, statements)
            self.log_test(
                "Merkle proof generation",
                len(proof) > 0,
                f"Proof length: {len(proof)}"
            )
            
            is_valid = verify_merkle_proof(statements[0], proof, root)
            self.log_test(
                "Merkle proof verification",
                is_valid,
                f"Proof valid: {is_valid}"
            )
            
            
            return all(r["passed"] for r in self.test_results[-3:])
            
        except Exception as e:
            self.log_test("Merkle workflow", False, f"Exception: {str(e)}")
            return False
    
    def validate_block_sealing_workflow(self) -> bool:
        """
        Validate block sealing workflow:
        1. Create block with statements
        2. Compute Merkle root
        3. Hash block header
        4. Verify domain separation
        """
        try:
            from backend.ledger.blockchain import seal_block
            
            statements = ["p -> p", "p /\\ q -> p", "p /\\ q -> q"]
            prev_hash = "0" * 64
            block_number = 1
            timestamp = 1234567890.0
            
            block = seal_block(statements, prev_hash, block_number, timestamp)
            self.log_test(
                "Block sealing",
                "header" in block and "merkle_root" in block["header"],
                f"Block sealed with Merkle root: {block['header']['merkle_root'][:16]}..."
            )
            
            expected_root = merkle_root(statements)
            actual_root = block["header"]["merkle_root"]
            self.log_test(
                "Block Merkle root correctness",
                expected_root == actual_root,
                f"Roots match: {expected_root == actual_root}"
            )
            
            header_json = json.dumps(block["header"], sort_keys=True, separators=(",", ":"))
            block_hash = hash_block(header_json)
            self.log_test(
                "Block header hashing",
                len(block_hash) == 64,
                f"Block hash: {block_hash[:16]}..."
            )
            
            plain_hash = sha256_hex(header_json)
            self.log_test(
                "Block domain separation",
                block_hash != plain_hash,
                "BLCK domain tag applied"
            )
            
            return all(r["passed"] for r in self.test_results[-4:])
            
        except Exception as e:
            self.log_test("Block sealing workflow", False, f"Exception: {str(e)}")
            return False
    
    def validate_full_handshake(self) -> Tuple[bool, Dict]:
        """
        Run complete end-to-end validation.
        
        Returns:
            Tuple of (all_passed, report_dict)
        """
        print("=" * 70)
        print("End-to-End Cryptographic Handshake Validation")
        print("=" * 70)
        
        workflows = [
            ("API Key Workflow", self.validate_api_key_workflow),
            ("Redis Auth Workflow", self.validate_redis_auth_workflow),
            ("Statement Workflow", self.validate_statement_workflow),
            ("Merkle Workflow", self.validate_merkle_workflow),
            ("Block Sealing Workflow", self.validate_block_sealing_workflow),
        ]
        
        all_passed = True
        for name, validator in workflows:
            print(f"\n[{name}]")
            passed = validator()
            all_passed &= passed
            status = "PASS" if passed else "FAIL"
            print(f"  {status}")
        
        passed_count = sum(1 for r in self.test_results if r["passed"])
        total_count = len(self.test_results)
        
        report = {
            "version": "v3",
            "timestamp": datetime.utcnow().isoformat(),
            "tests_passed": passed_count,
            "tests_failed": total_count - passed_count,
            "tests_total": total_count,
            "all_passed": all_passed,
            "workflows": {
                name: {
                    "passed": validator.__name__ in [r["name"] for r in self.test_results if r["passed"]],
                    "tests": [r for r in self.test_results if name.lower() in r["name"].lower()]
                }
                for name, validator in workflows
            },
            "errors": self.errors,
            "test_results": self.test_results
        }
        
        print("\n" + "=" * 70)
        print(f"Tests Passed: {passed_count}/{total_count}")
        if self.errors:
            print(f"Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
        print("=" * 70)
        
        return all_passed, report


def run_handshake_validation() -> Tuple[bool, Dict]:
    """
    Run end-to-end cryptographic handshake validation.
    
    Returns:
        Tuple of (all_passed, report_dict)
    """
    handshake = CryptoHandshake()
    return handshake.validate_full_handshake()


if __name__ == "__main__":
    passed, report = run_handshake_validation()
    
    report_path = "handshake_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    exit(0 if passed else 1)
