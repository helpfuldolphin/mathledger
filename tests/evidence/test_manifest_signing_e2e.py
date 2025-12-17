#!/usr/bin/env python3
"""
End-to-End Integration Tests for Manifest Signing

Verifies the complete signing workflow:
1. Generate a minimal evidence pack
2. Sign the manifest
3. Verify signature passes
4. Tamper with manifest (modify one byte)
5. Verify signature fails (fail-close)

These tests confirm that the signing system provides tamper detection.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def keypair(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a test Ed25519 keypair."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    keys_dir = tmp_path / "keys"
    keys_dir.mkdir()

    private_path = keys_dir / "test_private.pem"
    public_path = keys_dir / "test_public.pem"

    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)

    return private_path, public_path


@pytest.fixture
def minimal_evidence_pack(tmp_path: Path) -> Path:
    """
    Create a minimal evidence pack structure.

    Structure:
        evidence_pack/
            manifest.json
            data/
                sample.txt
    """
    pack_dir = tmp_path / "evidence_pack"
    pack_dir.mkdir()

    # Create sample data file
    data_dir = pack_dir / "data"
    data_dir.mkdir()

    sample_file = data_dir / "sample.txt"
    sample_content = b"This is sample evidence data for testing.\n"
    sample_file.write_bytes(sample_content)

    # Compute hash
    sample_hash = hashlib.sha256(sample_content).hexdigest()

    # Create manifest
    manifest = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "generated_at": "2025-12-17T00:00:00Z",
        "files": [
            {
                "path": "data/sample.txt",
                "sha256": sample_hash,
                "size_bytes": len(sample_content),
            }
        ],
    }

    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return pack_dir


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestManifestSigningE2E:
    """End-to-end tests for manifest signing workflow."""

    def test_e2e_sign_verify_tamper_workflow(
        self,
        keypair: tuple[Path, Path],
        minimal_evidence_pack: Path,
    ):
        """
        Full E2E test: sign, verify, tamper, verify fails.

        This test confirms fail-close behavior: any modification to the
        manifest after signing causes verification to fail.
        """
        private_path, public_path = keypair
        manifest_path = minimal_evidence_pack / "manifest.json"
        signature_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")

        # Step 1: Sign the manifest
        result = subprocess.run(
            [
                sys.executable,
                "scripts/sign_manifest.py",
                "--manifest", str(manifest_path),
                "--key", str(private_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Signing failed: {result.stderr}"
        assert signature_path.exists(), "Signature file not created"
        assert "Status: SIGNED" in result.stdout

        # Step 2: Verify signature passes
        result = subprocess.run(
            [
                sys.executable,
                "scripts/verify_manifest_signature.py",
                "--manifest", str(manifest_path),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Verification should pass: {result.stderr}"
        assert "Status: VERIFIED" in result.stdout

        # Step 3: Tamper with manifest (modify one byte)
        original_content = manifest_path.read_bytes()
        tampered_content = original_content.replace(b'"1.0.0"', b'"1.0.1"')
        assert tampered_content != original_content, "Tampering should change content"
        manifest_path.write_bytes(tampered_content)

        # Step 4: Verify signature fails (fail-close)
        result = subprocess.run(
            [
                sys.executable,
                "scripts/verify_manifest_signature.py",
                "--manifest", str(manifest_path),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Verification should fail after tampering"
        assert "Status: INVALID" in result.stdout

    def test_e2e_single_bit_flip_detected(
        self,
        keypair: tuple[Path, Path],
        minimal_evidence_pack: Path,
    ):
        """
        Test that even a single bit flip is detected.

        This confirms the cryptographic strength of the signature.
        """
        private_path, public_path = keypair
        manifest_path = minimal_evidence_pack / "manifest.json"
        signature_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")

        # Sign the manifest
        subprocess.run(
            [
                sys.executable,
                "scripts/sign_manifest.py",
                "--manifest", str(manifest_path),
                "--key", str(private_path),
            ],
            capture_output=True,
        )

        # Flip a single bit in the manifest
        original_bytes = bytearray(manifest_path.read_bytes())
        # Find a non-whitespace byte to flip
        for i, b in enumerate(original_bytes):
            if b not in (ord(' '), ord('\n'), ord('\r'), ord('\t')):
                original_bytes[i] ^= 0x01  # Flip lowest bit
                break
        manifest_path.write_bytes(bytes(original_bytes))

        # Verify should fail
        result = subprocess.run(
            [
                sys.executable,
                "scripts/verify_manifest_signature.py",
                "--manifest", str(manifest_path),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Single bit flip should be detected"
        assert "Status: INVALID" in result.stdout

    def test_e2e_missing_signature_fails(
        self,
        keypair: tuple[Path, Path],
        minimal_evidence_pack: Path,
    ):
        """
        Test that verification fails when signature file is missing.

        This confirms fail-close behavior for missing signatures.
        """
        _, public_path = keypair
        manifest_path = minimal_evidence_pack / "manifest.json"
        signature_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")

        # Ensure no signature exists
        if signature_path.exists():
            signature_path.unlink()

        # Verify should fail
        result = subprocess.run(
            [
                sys.executable,
                "scripts/verify_manifest_signature.py",
                "--manifest", str(manifest_path),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Missing signature should fail verification"
        assert "SIGNATURE_MISSING" in result.stdout or "not found" in result.stdout.lower()

    def test_e2e_signature_not_transferable(
        self,
        keypair: tuple[Path, Path],
        minimal_evidence_pack: Path,
        tmp_path: Path,
    ):
        """
        Test that a signature from one manifest cannot be used for another.

        This confirms the signature is bound to the specific manifest content.
        """
        private_path, public_path = keypair
        manifest_path = minimal_evidence_pack / "manifest.json"

        # Sign the original manifest
        subprocess.run(
            [
                sys.executable,
                "scripts/sign_manifest.py",
                "--manifest", str(manifest_path),
                "--key", str(private_path),
            ],
            capture_output=True,
        )

        signature_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")
        original_signature = signature_path.read_bytes()

        # Create a different manifest
        different_pack = tmp_path / "different_pack"
        different_pack.mkdir()
        different_manifest = different_pack / "manifest.json"
        different_manifest.write_text(json.dumps({
            "schema_version": "2.0.0",
            "mode": "DIFFERENT",
            "files": [],
        }))

        # Copy the original signature to the different manifest
        different_sig = different_manifest.with_suffix(different_manifest.suffix + ".sig")
        different_sig.write_bytes(original_signature)

        # Verify should fail - signature doesn't match different content
        result = subprocess.run(
            [
                sys.executable,
                "scripts/verify_manifest_signature.py",
                "--manifest", str(different_manifest),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Signature should not be transferable"
        assert "Status: INVALID" in result.stdout


class TestKeyGeneration:
    """Tests for key generation script."""

    def test_e2e_generate_keypair_via_script(self, tmp_path: Path):
        """Test keypair generation via the script."""
        keys_dir = tmp_path / "generated_keys"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_signing_keypair.py",
                "--output-dir", str(keys_dir),
                "--name", "test_gen",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Key generation failed: {result.stderr}"
        assert (keys_dir / "test_gen_private.pem").exists()
        assert (keys_dir / "test_gen_public.pem").exists()
        assert "SECURITY NOTICE" in result.stdout

    def test_e2e_generated_keypair_works_for_signing(self, tmp_path: Path):
        """Test that script-generated keys work for signing/verification."""
        keys_dir = tmp_path / "keys"

        # Generate keypair via script
        subprocess.run(
            [
                sys.executable,
                "scripts/generate_signing_keypair.py",
                "--output-dir", str(keys_dir),
                "--name", "workflow",
            ],
            capture_output=True,
        )

        private_path = keys_dir / "workflow_private.pem"
        public_path = keys_dir / "workflow_public.pem"

        # Create test manifest
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text('{"test": "data"}')

        # Sign
        result = subprocess.run(
            [
                sys.executable,
                "scripts/sign_manifest.py",
                "--manifest", str(manifest_path),
                "--key", str(private_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify
        result = subprocess.run(
            [
                sys.executable,
                "scripts/verify_manifest_signature.py",
                "--manifest", str(manifest_path),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Status: VERIFIED" in result.stdout


class TestFailClose:
    """Tests confirming fail-close security behavior."""

    def test_verification_fails_on_corrupted_signature(
        self,
        keypair: tuple[Path, Path],
        minimal_evidence_pack: Path,
    ):
        """Test that corrupted signature file causes verification failure."""
        private_path, public_path = keypair
        manifest_path = minimal_evidence_pack / "manifest.json"

        # Sign the manifest
        subprocess.run(
            [
                sys.executable,
                "scripts/sign_manifest.py",
                "--manifest", str(manifest_path),
                "--key", str(private_path),
            ],
            capture_output=True,
        )

        signature_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")

        # Corrupt the signature
        corrupted = bytearray(signature_path.read_bytes())
        corrupted[0] ^= 0xFF  # Flip all bits in first byte
        signature_path.write_bytes(bytes(corrupted))

        # Verify should fail
        result = subprocess.run(
            [
                sys.executable,
                "scripts/verify_manifest_signature.py",
                "--manifest", str(manifest_path),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, "Corrupted signature should fail"

    def test_verification_fails_with_truncated_signature(
        self,
        keypair: tuple[Path, Path],
        minimal_evidence_pack: Path,
    ):
        """Test that truncated signature file causes verification failure."""
        private_path, public_path = keypair
        manifest_path = minimal_evidence_pack / "manifest.json"

        # Sign the manifest
        subprocess.run(
            [
                sys.executable,
                "scripts/sign_manifest.py",
                "--manifest", str(manifest_path),
                "--key", str(private_path),
            ],
            capture_output=True,
        )

        signature_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")

        # Truncate the signature (Ed25519 signatures are 64 bytes)
        truncated = signature_path.read_bytes()[:32]
        signature_path.write_bytes(truncated)

        # Verify should fail
        result = subprocess.run(
            [
                sys.executable,
                "scripts/verify_manifest_signature.py",
                "--manifest", str(manifest_path),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Truncated signature should fail"
