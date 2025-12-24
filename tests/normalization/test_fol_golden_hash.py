"""
Golden hash guardrail for FOL_FIN_EQ_v1 evidence pack.

This test ensures no silent drift in the evidence pack generation.
Any change that alters the golden manifest hash requires a new Phase version.

Golden Manifest SHA256: 096ee79e4e20c94fffbc2ec9964dde98f8058cba47a887031085e0800d6d2113
"""

import hashlib
import subprocess
import sys

# The canonical golden hash from Phase 3 closure
GOLDEN_MANIFEST_SHA256 = "096ee79e4e20c94fffbc2ec9964dde98f8058cba47a887031085e0800d6d2113"


class TestGoldenHash:
    """Golden hash guardrail tests."""

    def test_demo_z2_manifest_matches_golden_hash(self, tmp_path):
        """Generated demo_z2 manifest must match the Phase 3 golden hash.

        This is the permanent 'no silent drift' alarm. If this test fails,
        either:
        1. A bug was introduced that breaks determinism, OR
        2. An intentional change was made that requires Phase 3.1

        See docs/FOL_FIN_EQ_PHASE3_CLOSURE.md for change policy.
        """
        output_dir = tmp_path / "demo_z2"

        # Run the demo
        result = subprocess.run(
            [
                sys.executable, "-m", "scripts.run_fol_fin_eq_demo",
                "--domain", "z2",
                "--output", str(output_dir)
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Demo failed: {result.stderr}"

        # Compute manifest hash
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists(), "manifest.json not created"

        manifest_bytes = manifest_path.read_bytes()
        actual_hash = hashlib.sha256(manifest_bytes).hexdigest()

        assert actual_hash == GOLDEN_MANIFEST_SHA256, (
            f"Manifest hash drift detected!\n"
            f"Expected: {GOLDEN_MANIFEST_SHA256}\n"
            f"Actual:   {actual_hash}\n"
            f"See docs/FOL_FIN_EQ_PHASE3_CLOSURE.md for change policy."
        )

    def test_verify_script_passes(self, tmp_path):
        """Generated verify.py must pass on its own output."""
        output_dir = tmp_path / "demo_z2"

        # Run the demo
        subprocess.run(
            [
                sys.executable, "-m", "scripts.run_fol_fin_eq_demo",
                "--domain", "z2",
                "--output", str(output_dir)
            ],
            capture_output=True,
            check=True
        )

        # Run verify.py
        result = subprocess.run(
            [sys.executable, str(output_dir / "verify.py")],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"verify.py failed: {result.stderr}"
        assert "PASS" in result.stdout
