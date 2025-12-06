#!/usr/bin/env python3
"""
Composite Dual Attestation (DA) CI Workflow

Reads UI and Reasoning merkle roots, generates composite DA token.
Implements:
- RFC8785 canonical JSON serialization
- ASCII-only output enforcement
- Fail-closed (ABSTAIN) on missing roots
- Proof-or-Abstain doctrine compliance

Input artifacts:
- artifacts/ui/roots.json
- artifacts/reasoning/roots.json

Output:
- UI_MERKLE_ROOT: <u_t>
- REASONING_MERKLE_ROOT: <r_t>
- COMPOSITE_DA_TOKEN: <H_t>
"""
import json
import hashlib
import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path


class RFC8785Canonicalizer:
    """
    RFC 8785 - JSON Canonicalization Scheme (JCS)
    Simplified implementation for deterministic JSON serialization
    """
    
    @staticmethod
    def canonicalize(obj: Any) -> str:
        """
        Canonicalize a Python object to RFC8785-compliant JSON string.
        
        Rules:
        1. No whitespace
        2. Keys sorted lexicographically
        3. Unicode escaping for non-ASCII
        4. No trailing zeros in numbers
        """
        return json.dumps(
            obj,
            ensure_ascii=True,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False
        )


class ASCIIGate:
    """Enforces ASCII-only output compliance"""
    
    @staticmethod
    def validate(text: str) -> bool:
        """Check if text is pure ASCII"""
        try:
            text.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False
    
    @staticmethod
    def sanitize(text: str) -> str:
        """Convert to ASCII, replacing non-ASCII with '?'"""
        return text.encode('ascii', errors='replace').decode('ascii')
    
    @staticmethod
    def enforce(text: str) -> str:
        """Enforce ASCII or raise error"""
        if not ASCIIGate.validate(text):
            raise ValueError(f"Non-ASCII content detected: {text[:50]}...")
        return text


class CompositeDualAttestation:
    """
    Composite Dual Attestation Token Generator
    
    Combines UI merkle root (u_t) and Reasoning merkle root (r_t)
    into a composite hash (H_t) for cryptographic attestation.
    """
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.ui_roots_path = self.artifacts_dir / "ui" / "roots.json"
        self.reasoning_roots_path = self.artifacts_dir / "reasoning" / "roots.json"
        self.canonicalizer = RFC8785Canonicalizer()
    
    def load_root(self, path: Path, root_type: str) -> Optional[str]:
        """
        Load merkle root from JSON file.
        Returns None if file missing or invalid (fail-closed).
        """
        if not path.exists():
            print(f"ABSTAIN: {root_type} root file not found: {path}", file=sys.stderr)
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract merkle root (support multiple field names)
            root = data.get('merkle_root') or data.get('root') or data.get('hash')
            
            if not root:
                print(f"ABSTAIN: No merkle root found in {path}", file=sys.stderr)
                return None
            
            # Validate ASCII
            if not ASCIIGate.validate(root):
                print(f"ABSTAIN: Non-ASCII merkle root in {path}", file=sys.stderr)
                return None
            
            # Validate hex format (64 chars for SHA-256)
            if not isinstance(root, str) or len(root) != 64:
                print(f"ABSTAIN: Invalid merkle root format in {path} (expected 64-char hex)", file=sys.stderr)
                return None
            
            try:
                int(root, 16)  # Verify it's valid hex
            except ValueError:
                print(f"ABSTAIN: Merkle root is not valid hex in {path}", file=sys.stderr)
                return None
            
            return root
        
        except json.JSONDecodeError as e:
            print(f"ABSTAIN: Invalid JSON in {path}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"ABSTAIN: Error reading {path}: {e}", file=sys.stderr)
            return None
    
    def generate_composite_token(self, ui_root: str, reasoning_root: str) -> str:
        """
        Generate composite DA token H_t = SHA256(canonical({u_t, r_t}))
        
        Uses RFC8785 canonicalization for deterministic hashing.
        """
        composite_data = {
            "ui_merkle_root": ui_root,
            "reasoning_merkle_root": reasoning_root,
            "version": "v1",
            "algorithm": "SHA256",
            "canonicalization": "RFC8785"
        }
        
        # Canonicalize
        canonical_json = self.canonicalizer.canonicalize(composite_data)
        
        # Enforce ASCII
        canonical_json = ASCIIGate.enforce(canonical_json)
        
        # Hash
        composite_hash = hashlib.sha256(canonical_json.encode('ascii')).hexdigest()
        
        return composite_hash
    
    def run(self) -> int:
        """
        Execute composite DA workflow.
        
        Returns:
            0 on success (PASS)
            1 on failure (ABSTAIN)
        """
        print("=" * 80)
        print("COMPOSITE DUAL ATTESTATION (DA) WORKFLOW")
        print("=" * 80)
        print()
        
        # Load UI root
        ui_root = self.load_root(self.ui_roots_path, "UI")
        
        # Load Reasoning root
        reasoning_root = self.load_root(self.reasoning_roots_path, "Reasoning")
        
        # Fail-closed: ABSTAIN if either root is missing
        if ui_root is None or reasoning_root is None:
            print()
            print("=" * 80)
            print("VERDICT: ABSTAIN")
            print("=" * 80)
            print("Reason: Missing or invalid merkle roots (fail-closed)")
            print()
            print("UI_MERKLE_ROOT: MISSING" if ui_root is None else f"UI_MERKLE_ROOT: {ui_root}")
            print("REASONING_MERKLE_ROOT: MISSING" if reasoning_root is None else f"REASONING_MERKLE_ROOT: {reasoning_root}")
            print("COMPOSITE_DA_TOKEN: ABSTAIN")
            return 1
        
        # Generate composite token
        try:
            composite_token = self.generate_composite_token(ui_root, reasoning_root)
        except Exception as e:
            print(f"ABSTAIN: Error generating composite token: {e}", file=sys.stderr)
            print()
            print("=" * 80)
            print("VERDICT: ABSTAIN")
            print("=" * 80)
            print(f"UI_MERKLE_ROOT: {ui_root}")
            print(f"REASONING_MERKLE_ROOT: {reasoning_root}")
            print("COMPOSITE_DA_TOKEN: ERROR")
            return 1
        
        # Success: PASS
        print("✓ UI merkle root loaded")
        print("✓ Reasoning merkle root loaded")
        print("✓ Composite DA token generated")
        print()
        print("=" * 80)
        print("VERDICT: PASS")
        print("=" * 80)
        print(f"UI_MERKLE_ROOT: {ui_root}")
        print(f"REASONING_MERKLE_ROOT: {reasoning_root}")
        print(f"COMPOSITE_DA_TOKEN: {composite_token}")
        print()
        print("Canonicalization: RFC8785")
        print("ASCII Compliance: ENFORCED")
        print("Fail-Closed: ACTIVE")
        
        return 0


def create_mock_roots():
    """Create mock root files for testing"""
    artifacts_dir = Path("artifacts")
    
    # Create directories
    (artifacts_dir / "ui").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "reasoning").mkdir(parents=True, exist_ok=True)
    
    # Create mock UI root
    ui_root = {
        "merkle_root": hashlib.sha256(b"mock_ui_data").hexdigest(),
        "timestamp": "2025-10-19T15:50:00Z",
        "version": "v1"
    }
    
    with open(artifacts_dir / "ui" / "roots.json", 'w') as f:
        json.dump(ui_root, f, indent=2)
    
    # Create mock Reasoning root (from proof simulator)
    # Use the last merkle from the proof generation
    metrics_file = artifacts_dir / "wpv5" / "run_metrics_v1.jsonl"
    
    if metrics_file.exists():
        # Read last line
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_metric = json.loads(lines[-1])
                reasoning_root = {
                    "merkle_root": last_metric['merkle'],
                    "block_no": last_metric['block_no'],
                    "inserted_proofs": last_metric['inserted_proofs'],
                    "timestamp": "2025-10-19T15:50:00Z",
                    "version": "v1"
                }
            else:
                reasoning_root = {
                    "merkle_root": hashlib.sha256(b"mock_reasoning_data").hexdigest(),
                    "timestamp": "2025-10-19T15:50:00Z",
                    "version": "v1"
                }
    else:
        reasoning_root = {
            "merkle_root": hashlib.sha256(b"mock_reasoning_data").hexdigest(),
            "timestamp": "2025-10-19T15:50:00Z",
            "version": "v1"
        }
    
    with open(artifacts_dir / "reasoning" / "roots.json", 'w') as f:
        json.dump(reasoning_root, f, indent=2)
    
    print("✓ Mock root files created")
    print(f"  - {artifacts_dir / 'ui' / 'roots.json'}")
    print(f"  - {artifacts_dir / 'reasoning' / 'roots.json'}")
    print()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Composite Dual Attestation CI Workflow")
    parser.add_argument('--artifacts-dir', default='artifacts', help='Artifacts directory path')
    parser.add_argument('--create-mocks', action='store_true', help='Create mock root files for testing')
    
    args = parser.parse_args()
    
    # Create mock roots if requested
    if args.create_mocks:
        create_mock_roots()
    
    # Run composite DA workflow
    da = CompositeDualAttestation(artifacts_dir=args.artifacts_dir)
    return da.run()


if __name__ == "__main__":
    raise SystemExit(main())

