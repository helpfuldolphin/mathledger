#!/usr/bin/env python3
"""
Manifest Signature Verification Script

Verifies a detached Ed25519 signature for evidence pack manifests.

SHADOW MODE CONTRACT:
- This script is purely observational; it never modifies files
- Uses standard exit codes for automation
- Outputs neutral, non-evaluative messages

Usage:
    python scripts/verify_manifest_signature.py --manifest path/to/manifest.json --pubkey path/to/public.pem
    python scripts/verify_manifest_signature.py --manifest path/to/manifest.json --pubkey path/to/public.pem --signature manifest.json.sig

Exit Codes:
    0 = Signature verified successfully
    1 = Signature verification failed (invalid signature, tampered manifest)
    2 = Configuration error (missing files, invalid key format)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def load_public_key(key_path: Path) -> Ed25519PublicKey:
    """
    Load an Ed25519 public key from PEM file.

    Args:
        key_path: Path to PEM-encoded public key

    Returns:
        Ed25519PublicKey instance

    Raises:
        ValueError: If key format is invalid
        FileNotFoundError: If key file does not exist
    """
    with open(key_path, "rb") as f:
        key_data = f.read()

    public_key = serialization.load_pem_public_key(key_data)

    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("Key must be Ed25519 public key")

    return public_key


def verify_signature(
    file_path: Path,
    signature_path: Path,
    public_key: Ed25519PublicKey,
) -> bool:
    """
    Verify Ed25519 signature for a file.

    Args:
        file_path: Path to file that was signed
        signature_path: Path to detached signature file
        public_key: Ed25519 public key

    Returns:
        True if signature is valid, False otherwise
    """
    with open(file_path, "rb") as f:
        data = f.read()

    with open(signature_path, "rb") as f:
        signature = f.read()

    try:
        public_key.verify(signature, data)
        return True
    except InvalidSignature:
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify detached Ed25519 signature for evidence pack manifest"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest.json file to verify",
    )
    parser.add_argument(
        "--pubkey",
        type=str,
        required=True,
        help="Path to Ed25519 public key (PEM format)",
    )
    parser.add_argument(
        "--signature",
        type=str,
        help="Path to signature file (default: <manifest>.sig)",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    pubkey_path = Path(args.pubkey)

    # Determine signature path
    if args.signature:
        signature_path = Path(args.signature)
    else:
        signature_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")

    # Validate manifest exists
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    # Validate public key exists
    if not pubkey_path.exists():
        print(f"Public key not found: {pubkey_path}", file=sys.stderr)
        return 2

    # Validate signature exists
    if not signature_path.exists():
        print(f"Signature not found: {signature_path}", file=sys.stderr)
        print("Status: SIGNATURE_MISSING")
        return 1

    # Load public key
    try:
        public_key = load_public_key(pubkey_path)
    except ValueError as e:
        print(f"Invalid public key: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Failed to load public key: {e}", file=sys.stderr)
        return 2

    # Verify signature
    try:
        is_valid = verify_signature(manifest_path, signature_path, public_key)
    except Exception as e:
        print(f"Verification error: {e}", file=sys.stderr)
        return 2

    # Output result
    print(f"Manifest: {manifest_path}")
    print(f"Signature: {signature_path}")
    print(f"Public key: {pubkey_path}")

    if is_valid:
        print("Status: VERIFIED")
        return 0
    else:
        print("Status: INVALID")
        print("The signature does not match the manifest content.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
