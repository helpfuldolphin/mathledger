#!/usr/bin/env python3
"""
Manifest Signing Script

Creates a detached Ed25519 signature for evidence pack manifests.

SHADOW MODE CONTRACT:
- This script produces a signature file; it never modifies the manifest
- Private key material is never logged or printed
- Uses standard exit codes

Usage:
    python scripts/sign_manifest.py --manifest path/to/manifest.json --key path/to/private.pem
    python scripts/sign_manifest.py --manifest path/to/manifest.json --key path/to/private.pem --output manifest.json.sig

Exit Codes:
    0 = Signature created successfully
    1 = Signing failed (missing files, invalid key, etc.)
    2 = Configuration error (missing arguments)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def load_private_key(key_path: Path) -> Ed25519PrivateKey:
    """
    Load an Ed25519 private key from PEM file.

    Args:
        key_path: Path to PEM-encoded private key

    Returns:
        Ed25519PrivateKey instance

    Raises:
        ValueError: If key format is invalid
        FileNotFoundError: If key file does not exist
    """
    with open(key_path, "rb") as f:
        key_data = f.read()

    private_key = serialization.load_pem_private_key(key_data, password=None)

    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("Key must be Ed25519 private key")

    return private_key


def sign_file(file_path: Path, private_key: Ed25519PrivateKey) -> bytes:
    """
    Create Ed25519 signature for a file.

    Args:
        file_path: Path to file to sign
        private_key: Ed25519 private key

    Returns:
        64-byte signature
    """
    with open(file_path, "rb") as f:
        data = f.read()

    return private_key.sign(data)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create detached Ed25519 signature for evidence pack manifest"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest.json file to sign",
    )
    parser.add_argument(
        "--key",
        type=str,
        required=True,
        help="Path to Ed25519 private key (PEM format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for signature file (default: <manifest>.sig)",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    key_path = Path(args.key)

    # Validate manifest exists
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    # Validate key exists
    if not key_path.exists():
        print(f"Private key not found: {key_path}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")

    # Load private key
    try:
        private_key = load_private_key(key_path)
    except ValueError as e:
        print(f"Invalid private key: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed to load private key: {e}", file=sys.stderr)
        return 1

    # Sign manifest
    try:
        signature = sign_file(manifest_path, private_key)
    except Exception as e:
        print(f"Signing failed: {e}", file=sys.stderr)
        return 1

    # Write signature (raw bytes)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(signature)
    except Exception as e:
        print(f"Failed to write signature: {e}", file=sys.stderr)
        return 1

    # Output confirmation (no sensitive data)
    print(f"Manifest: {manifest_path}")
    print(f"Signature: {output_path}")
    print(f"Signature size: {len(signature)} bytes")
    print("Status: SIGNED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
