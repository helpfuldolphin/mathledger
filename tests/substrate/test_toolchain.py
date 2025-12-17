"""
Unit tests for substrate.repro.toolchain module.

SAVE TO REPO: YES
Rationale: Verifies toolchain fingerprint computation is deterministic.
"""

import json
import tempfile
from pathlib import Path

import pytest

from substrate.repro.toolchain import (
    ToolchainSnapshot,
    PythonToolchain,
    LeanToolchain,
    PlatformInfo,
    capture_toolchain_snapshot,
    compute_toolchain_fingerprint,
    verify_toolchain_match,
    save_toolchain_snapshot,
    load_toolchain_snapshot,
)


class TestComputeToolchainFingerprint:
    """Tests for compute_toolchain_fingerprint function."""

    def test_deterministic(self):
        """Same inputs produce same fingerprint."""
        fp1 = compute_toolchain_fingerprint("a", "b", "c", "d")
        fp2 = compute_toolchain_fingerprint("a", "b", "c", "d")
        assert fp1 == fp2

    def test_different_inputs_different_fingerprint(self):
        """Different inputs produce different fingerprints."""
        fp1 = compute_toolchain_fingerprint("a", "b", "c", "d")
        fp2 = compute_toolchain_fingerprint("a", "b", "c", "e")
        assert fp1 != fp2

    def test_order_matters(self):
        """Order of hashes affects fingerprint."""
        fp1 = compute_toolchain_fingerprint("a", "b", "c", "d")
        fp2 = compute_toolchain_fingerprint("b", "a", "c", "d")
        assert fp1 != fp2

    def test_returns_hex_sha256(self):
        """Fingerprint is 64-char hex string (SHA-256)."""
        fp = compute_toolchain_fingerprint("a", "b", "c", "d")
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


class TestCaptureToolchainSnapshot:
    """Tests for capture_toolchain_snapshot function."""

    def test_captures_snapshot(self):
        """Can capture a valid snapshot from repo root."""
        # Find repo root
        current = Path(__file__).resolve()
        repo_root = None
        for parent in current.parents:
            if (parent / "uv.lock").exists():
                repo_root = parent
                break

        if repo_root is None:
            pytest.skip("Cannot find repo root")

        snapshot = capture_toolchain_snapshot(repo_root)

        assert snapshot.schema_version == "1.0"
        assert len(snapshot.fingerprint) == 64
        assert snapshot.python.version
        assert snapshot.lean.version
        assert snapshot.platform.os

    def test_deterministic(self):
        """Same repo root produces same fingerprint."""
        current = Path(__file__).resolve()
        repo_root = None
        for parent in current.parents:
            if (parent / "uv.lock").exists():
                repo_root = parent
                break

        if repo_root is None:
            pytest.skip("Cannot find repo root")

        snap1 = capture_toolchain_snapshot(repo_root)
        snap2 = capture_toolchain_snapshot(repo_root)

        assert snap1.fingerprint == snap2.fingerprint
        assert snap1.python.uv_lock_hash == snap2.python.uv_lock_hash
        assert snap1.lean.lake_manifest_hash == snap2.lean.lake_manifest_hash


class TestVerifyToolchainMatch:
    """Tests for verify_toolchain_match function."""

    def test_identical_snapshots_match(self):
        """Identical snapshots should match."""
        python = PythonToolchain("3.11.9", "0.8.16", "abc123")
        lean = LeanToolchain("v4.23.0", "def456", "ghi789", "jkl012")
        platform = PlatformInfo("Windows-10", "AMD64", "test-host")

        snap1 = ToolchainSnapshot("1.0", "fingerprint123", python, lean, platform)
        snap2 = ToolchainSnapshot("1.0", "fingerprint123", python, lean, platform)

        match, diffs = verify_toolchain_match(snap1, snap2)
        assert match
        assert len(diffs) == 0

    def test_different_fingerprints_mismatch(self):
        """Different fingerprints should mismatch."""
        python = PythonToolchain("3.11.9", "0.8.16", "abc123")
        lean = LeanToolchain("v4.23.0", "def456", "ghi789", "jkl012")
        platform = PlatformInfo("Windows-10", "AMD64", "test-host")

        snap1 = ToolchainSnapshot("1.0", "fingerprint123", python, lean, platform)
        snap2 = ToolchainSnapshot("1.0", "fingerprint456", python, lean, platform)

        match, diffs = verify_toolchain_match(snap1, snap2)
        assert not match
        assert len(diffs) > 0
        assert any("Fingerprint" in d for d in diffs)

    def test_strict_mode_checks_all_hashes(self):
        """Strict mode should check individual hashes."""
        python1 = PythonToolchain("3.11.9", "0.8.16", "abc123")
        python2 = PythonToolchain("3.11.9", "0.8.16", "different")
        lean = LeanToolchain("v4.23.0", "def456", "ghi789", "jkl012")
        platform = PlatformInfo("Windows-10", "AMD64", "test-host")

        snap1 = ToolchainSnapshot("1.0", "fp", python1, lean, platform)
        snap2 = ToolchainSnapshot("1.0", "fp", python2, lean, platform)

        match, diffs = verify_toolchain_match(snap1, snap2, strict=True)
        assert not match
        assert any("uv.lock" in d for d in diffs)

    def test_non_strict_mode_only_checks_fingerprint(self):
        """Non-strict mode should only check fingerprint."""
        python1 = PythonToolchain("3.11.9", "0.8.16", "abc123")
        python2 = PythonToolchain("3.11.9", "0.8.16", "different")
        lean = LeanToolchain("v4.23.0", "def456", "ghi789", "jkl012")
        platform = PlatformInfo("Windows-10", "AMD64", "test-host")

        snap1 = ToolchainSnapshot("1.0", "same_fp", python1, lean, platform)
        snap2 = ToolchainSnapshot("1.0", "same_fp", python2, lean, platform)

        match, diffs = verify_toolchain_match(snap1, snap2, strict=False)
        assert match
        assert len(diffs) == 0


class TestSaveLoadToolchainSnapshot:
    """Tests for save/load functions."""

    def test_round_trip(self):
        """Snapshot should survive save/load round trip."""
        python = PythonToolchain("3.11.9", "0.8.16", "abc123")
        lean = LeanToolchain("v4.23.0", "def456", "ghi789", "jkl012")
        platform = PlatformInfo("Windows-10", "AMD64", "test-host")
        original = ToolchainSnapshot("1.0", "fingerprint123", python, lean, platform)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_toolchain_snapshot(original, path)
            loaded = load_toolchain_snapshot(path)

            assert loaded.schema_version == original.schema_version
            assert loaded.fingerprint == original.fingerprint
            assert loaded.python.version == original.python.version
            assert loaded.lean.version == original.lean.version
            assert loaded.platform.hostname == original.platform.hostname
        finally:
            path.unlink(missing_ok=True)

    def test_to_dict(self):
        """to_dict should produce valid JSON-serializable dict."""
        python = PythonToolchain("3.11.9", "0.8.16", "abc123")
        lean = LeanToolchain("v4.23.0", "def456", "ghi789", "jkl012")
        platform = PlatformInfo("Windows-10", "AMD64", "test-host")
        snapshot = ToolchainSnapshot("1.0", "fingerprint123", python, lean, platform)

        d = snapshot.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert json_str

        # Should have expected structure
        assert d["schema_version"] == "1.0"
        assert d["fingerprint"] == "fingerprint123"
        assert d["python"]["version"] == "3.11.9"
        assert d["lean"]["version"] == "v4.23.0"
        assert d["platform"]["hostname"] == "test-host"
