#!/usr/bin/env python3
"""
Tests for manifest signing and verification.

Tests the Ed25519 signature workflow:
- Signing produces valid signature
- Verification passes for valid signature
- Verification fails if manifest changes post-signing
- Verification fails if signature is missing
"""

from __future__ import annotations

import json
import tempfile
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

    private_path = tmp_path / "test_private.pem"
    public_path = tmp_path / "test_public.pem"

    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)

    return private_path, public_path


@pytest.fixture
def sample_manifest(tmp_path: Path) -> Path:
    """Create a sample manifest.json."""
    manifest = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "files": [
            {"path": "test.txt", "sha256": "abc123"},
        ],
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest_path


# ============================================================================
# Import Tests (verify modules load correctly)
# ============================================================================

def test_sign_manifest_imports():
    """Verify sign_manifest module imports successfully."""
    from scripts.sign_manifest import load_private_key, sign_file
    assert callable(load_private_key)
    assert callable(sign_file)


def test_verify_manifest_signature_imports():
    """Verify verify_manifest_signature module imports successfully."""
    from scripts.verify_manifest_signature import load_public_key, verify_signature
    assert callable(load_public_key)
    assert callable(verify_signature)


# ============================================================================
# Signing Tests
# ============================================================================

def test_sign_manifest_produces_signature(keypair: tuple[Path, Path], sample_manifest: Path):
    """Signing a manifest produces a 64-byte Ed25519 signature."""
    from scripts.sign_manifest import load_private_key, sign_file

    private_path, _ = keypair

    private_key = load_private_key(private_path)
    signature = sign_file(sample_manifest, private_key)

    assert isinstance(signature, bytes)
    assert len(signature) == 64  # Ed25519 signatures are always 64 bytes


def test_sign_manifest_deterministic(keypair: tuple[Path, Path], sample_manifest: Path):
    """Signing the same manifest produces the same signature."""
    from scripts.sign_manifest import load_private_key, sign_file

    private_path, _ = keypair

    private_key = load_private_key(private_path)
    sig1 = sign_file(sample_manifest, private_key)
    sig2 = sign_file(sample_manifest, private_key)

    assert sig1 == sig2


# ============================================================================
# Verification Tests
# ============================================================================

def test_verification_passes_for_valid_signature(
    keypair: tuple[Path, Path],
    sample_manifest: Path,
):
    """Verification passes when signature matches manifest."""
    from scripts.sign_manifest import load_private_key, sign_file
    from scripts.verify_manifest_signature import load_public_key, verify_signature

    private_path, public_path = keypair

    # Sign
    private_key = load_private_key(private_path)
    signature = sign_file(sample_manifest, private_key)

    signature_path = sample_manifest.with_suffix(sample_manifest.suffix + ".sig")
    signature_path.write_bytes(signature)

    # Verify
    public_key = load_public_key(public_path)
    is_valid = verify_signature(sample_manifest, signature_path, public_key)

    assert is_valid is True


def test_verification_fails_if_manifest_changes(
    keypair: tuple[Path, Path],
    sample_manifest: Path,
):
    """Verification fails when manifest is modified after signing."""
    from scripts.sign_manifest import load_private_key, sign_file
    from scripts.verify_manifest_signature import load_public_key, verify_signature

    private_path, public_path = keypair

    # Sign original manifest
    private_key = load_private_key(private_path)
    signature = sign_file(sample_manifest, private_key)

    signature_path = sample_manifest.with_suffix(sample_manifest.suffix + ".sig")
    signature_path.write_bytes(signature)

    # Modify manifest after signing
    manifest_data = json.loads(sample_manifest.read_text())
    manifest_data["tampered"] = True
    sample_manifest.write_text(json.dumps(manifest_data, indent=2))

    # Verify should fail
    public_key = load_public_key(public_path)
    is_valid = verify_signature(sample_manifest, signature_path, public_key)

    assert is_valid is False


def test_verification_fails_if_signature_missing(
    keypair: tuple[Path, Path],
    sample_manifest: Path,
):
    """Verification fails when signature file does not exist."""
    from scripts.verify_manifest_signature import load_public_key, verify_signature

    _, public_path = keypair

    # No signature file created
    signature_path = sample_manifest.with_suffix(sample_manifest.suffix + ".sig")
    assert not signature_path.exists()

    public_key = load_public_key(public_path)

    # verify_signature should return False or raise when file missing
    # The implementation returns False via exception handling
    try:
        is_valid = verify_signature(sample_manifest, signature_path, public_key)
        # If it returns, it should be False
        assert is_valid is False
    except FileNotFoundError:
        # Also acceptable
        pass


def test_verification_fails_with_wrong_key(
    keypair: tuple[Path, Path],
    sample_manifest: Path,
    tmp_path: Path,
):
    """Verification fails when using a different public key."""
    from scripts.sign_manifest import load_private_key, sign_file
    from scripts.verify_manifest_signature import load_public_key, verify_signature

    private_path, _ = keypair

    # Sign with original key
    private_key = load_private_key(private_path)
    signature = sign_file(sample_manifest, private_key)

    signature_path = sample_manifest.with_suffix(sample_manifest.suffix + ".sig")
    signature_path.write_bytes(signature)

    # Generate a different keypair
    different_private = Ed25519PrivateKey.generate()
    different_public = different_private.public_key()

    different_public_pem = different_public.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    different_public_path = tmp_path / "different_public.pem"
    different_public_path.write_bytes(different_public_pem)

    # Verify with wrong key should fail
    wrong_public_key = load_public_key(different_public_path)
    is_valid = verify_signature(sample_manifest, signature_path, wrong_public_key)

    assert is_valid is False


# ============================================================================
# Key Loading Tests
# ============================================================================

def test_load_private_key_rejects_public_key(keypair: tuple[Path, Path]):
    """load_private_key raises ValueError when given a public key."""
    from scripts.sign_manifest import load_private_key

    _, public_path = keypair

    with pytest.raises(ValueError):
        load_private_key(public_path)


def test_load_public_key_rejects_private_key(keypair: tuple[Path, Path]):
    """load_public_key raises ValueError when given a private key."""
    from scripts.verify_manifest_signature import load_public_key

    private_path, _ = keypair

    with pytest.raises((ValueError, TypeError)):
        load_public_key(private_path)


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_sign_verify_workflow(keypair: tuple[Path, Path], sample_manifest: Path):
    """Full workflow: generate, sign, verify succeeds."""
    from scripts.sign_manifest import load_private_key, sign_file
    from scripts.verify_manifest_signature import load_public_key, verify_signature

    private_path, public_path = keypair

    # Step 1: Sign
    private_key = load_private_key(private_path)
    signature = sign_file(sample_manifest, private_key)

    signature_path = sample_manifest.with_suffix(sample_manifest.suffix + ".sig")
    signature_path.write_bytes(signature)

    # Step 2: Verify
    public_key = load_public_key(public_path)
    is_valid = verify_signature(sample_manifest, signature_path, public_key)

    assert is_valid is True

    # Step 3: Tamper and verify fails
    manifest_data = json.loads(sample_manifest.read_text())
    manifest_data["extra_field"] = "tampered"
    sample_manifest.write_text(json.dumps(manifest_data, indent=2))

    is_valid_after_tamper = verify_signature(sample_manifest, signature_path, public_key)
    assert is_valid_after_tamper is False
