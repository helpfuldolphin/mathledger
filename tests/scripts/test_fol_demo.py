"""
Integration tests for FOL demo script and output determinism.

Phase 2 RED: These tests will fail until scripts/run_fol_fin_eq_demo.py is implemented.

Tests demo integration including:
- Z2 verification passes
- Output determinism (byte-identical manifests)
- verify.py passes
- Broken domain produces REFUTED
- Large domain produces ABSTAINED
- Artifacts sorted by ID for determinism
"""

import pytest
import subprocess
import json
from pathlib import Path
import hashlib

# Import module that doesn't exist yet (will fail with ModuleNotFoundError)
from scripts.run_fol_fin_eq_demo import run_demo, generate_manifest


class TestFolDemo:
    """Integration tests for run_fol_fin_eq_demo.py."""

    @pytest.fixture
    def demo_z2_output(self, tmp_path):
        """Run demo on Z2 domain and return output path."""
        output_dir = tmp_path / "demo_z2"
        subprocess.run([
            "uv", "run", "python", "scripts/run_fol_fin_eq_demo.py",
            "--domain", "z2", "--output", str(output_dir)
        ], check=True)
        return output_dir

    def test_z2_verification_passes(self, demo_z2_output):
        """Z2 group axioms verify successfully."""
        certs = list((demo_z2_output / "certificates").glob("*.json"))
        assert len(certs) > 0
        for cert_path in certs:
            cert = json.loads(cert_path.read_text())
            assert cert["status"] == "VERIFIED"

    def test_output_determinism(self, tmp_path):
        """Two identical runs produce byte-identical manifest files."""
        output_a = tmp_path / "demo_a"
        output_b = tmp_path / "demo_b"
        subprocess.run([
            "uv", "run", "python", "scripts/run_fol_fin_eq_demo.py",
            "--domain", "z2", "--output", str(output_a)
        ], check=True)
        subprocess.run([
            "uv", "run", "python", "scripts/run_fol_fin_eq_demo.py",
            "--domain", "z2", "--output", str(output_b)
        ], check=True)
        manifest_a = (output_a / "manifest.json").read_bytes()
        manifest_b = (output_b / "manifest.json").read_bytes()
        assert manifest_a == manifest_b  # Byte-identical

    def test_verify_script_passes(self, demo_z2_output):
        """Generated verify.py passes with exit code 0."""
        result = subprocess.run(
            ["python", str(demo_z2_output / "verify.py")],
            capture_output=True
        )
        assert result.returncode == 0
        assert b"PASS" in result.stdout

    def test_broken_associativity_refuted(self, tmp_path):
        """Broken Z2 domain produces REFUTED certificate."""
        output_dir = tmp_path / "demo_broken"
        subprocess.run([
            "uv", "run", "python", "scripts/run_fol_fin_eq_demo.py",
            "--domain", "z2_broken", "--output", str(output_dir)
        ], check=True)
        certs = list((output_dir / "certificates").glob("*.json"))
        statuses = [json.loads(c.read_text())["status"] for c in certs]
        assert "REFUTED" in statuses

    def test_large_domain_abstains(self, tmp_path):
        """Large domain (100 elements) produces ABSTAINED certificate."""
        output_dir = tmp_path / "demo_large"
        subprocess.run([
            "uv", "run", "python", "scripts/run_fol_fin_eq_demo.py",
            "--domain", "large_100", "--output", str(output_dir)
        ], check=True)
        certs = list((output_dir / "certificates").glob("*.json"))
        statuses = [json.loads(c.read_text())["status"] for c in certs]
        assert "ABSTAINED" in statuses

    def test_manifest_artifacts_sorted_by_id(self, demo_z2_output):
        """Manifest artifacts array is sorted by artifact_id for determinism.

        This prevents diff failures caused by non-deterministic artifact ordering.
        """
        manifest = json.loads((demo_z2_output / "manifest.json").read_text())
        artifacts = manifest.get("artifacts", [])
        if len(artifacts) > 1:
            artifact_ids = [a["artifact_id"] for a in artifacts]
            assert artifact_ids == sorted(artifact_ids), "Artifacts must be sorted by artifact_id"
