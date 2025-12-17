#!/usr/bin/env python3
"""
Generate Ed25519 Signing Keypair

Creates a new Ed25519 keypair for manifest signing.

SECURITY WARNING:
- The private key grants signing authority
- NEVER commit private keys to version control
- Store private keys securely (e.g., secrets manager, encrypted vault)

Usage:
    python scripts/generate_signing_keypair.py --output-dir keys/
    python scripts/generate_signing_keypair.py --output-dir keys/ --name dev

Output Files:
    <output-dir>/<name>_private.pem  - Private key (KEEP SECRET)
    <output-dir>/<name>_public.pem   - Public key (safe to distribute)

Exit Codes:
    0 = Keypair generated successfully
    1 = Generation failed
    2 = Configuration error
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def generate_keypair(output_dir: Path, name: str) -> tuple[Path, Path]:
    """
    Generate Ed25519 keypair and save to PEM files.

    Args:
        output_dir: Directory for key files
        name: Key name prefix

    Returns:
        Tuple of (private_key_path, public_key_path)
    """
    # Generate keypair
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    # Serialize keys
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write private key with restricted permissions
    private_path = output_dir / f"{name}_private.pem"
    with open(private_path, "wb") as f:
        f.write(private_pem)

    # Set restrictive permissions on private key (Unix only)
    try:
        os.chmod(private_path, 0o600)
    except (OSError, AttributeError):
        pass  # Windows or permission error

    # Write public key
    public_path = output_dir / f"{name}_public.pem"
    with open(public_path, "wb") as f:
        f.write(public_pem)

    return private_path, public_path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Ed25519 keypair for manifest signing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for key files",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="manifest_signing",
        help="Key name prefix (default: manifest_signing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing keys",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    name = args.name

    # Check for existing keys
    private_path = output_dir / f"{name}_private.pem"
    public_path = output_dir / f"{name}_public.pem"

    if (private_path.exists() or public_path.exists()) and not args.force:
        print("Keys already exist. Use --force to overwrite.", file=sys.stderr)
        print(f"  Private: {private_path}", file=sys.stderr)
        print(f"  Public: {public_path}", file=sys.stderr)
        return 2

    # Generate keypair
    try:
        private_path, public_path = generate_keypair(output_dir, name)
    except Exception as e:
        print(f"Failed to generate keypair: {e}", file=sys.stderr)
        return 1

    print("Ed25519 keypair generated successfully.")
    print()
    print(f"Private key: {private_path}")
    print(f"Public key:  {public_path}")
    print()
    print("SECURITY NOTICE:")
    print("  - NEVER commit the private key to version control")
    print("  - Store the private key securely")
    print("  - The public key may be committed or distributed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
