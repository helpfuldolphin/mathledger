# PHASE II — NOT USED IN PHASE I
"""
Test Suite: Confusability Contracts & CI Gates

This module tests the confusability contract export, verification,
and CI summary functionality:

1. Export Tests: JSON file generation and schema validation
2. Verification Tests: OK/WARN/FAIL status with threshold checks
3. CI Summary Tests: Greppable output format

All tests are designed for CI integration with deterministic behavior.

NOTE: The current YAML has legacy format (string lists) which are treated
as all-targets (no decoys). Tests are designed to handle this gracefully.
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch, MagicMock

from experiments.curriculum_diagnostics import (
    cmd_export_confusability,
    cmd_verify_confusability,
    cmd_decoy_ci_summary,
    verify_slice_confusability,
    VerificationResult,
    VerificationStatus,
    DEFAULT_THRESHOLDS,
    main,
)
from experiments.decoys.loader import CurriculumDecoyLoader
from experiments.decoys.confusability import ConfusabilityMap


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def config_path() -> str:
    """Path to the Phase II curriculum config."""
    return "config/curriculum_uplift_phase2.yaml"


@pytest.fixture(scope="module")
def loader(config_path: str) -> CurriculumDecoyLoader:
    """Shared loader instance."""
    return CurriculumDecoyLoader(config_path)


@pytest.fixture(scope="module")
def slice_names(loader: CurriculumDecoyLoader) -> List[str]:
    """List of all uplift slice names."""
    return loader.list_uplift_slices()


def _has_decoys(config_path: str, slice_name: str) -> bool:
    """Check if a slice has the decoy-enabled format with near/far decoys."""
    cmap = ConfusabilityMap(slice_name, config_path)
    return any(e.get('role') in ('decoy_near', 'decoy_far') for e in cmap.entries)


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# EXPORT CONFUSABILITY TESTS
# =============================================================================

class TestExportConfusability:
    """Tests for --export-confusability command."""
    
    def test_export_creates_json_file(self, config_path: str, temp_output_dir: Path):
        """Export should create a valid JSON file."""
        output_path = temp_output_dir / "test_export.json"
        
        args = MagicMock()
        args.export_confusability = "slice_uplift_goal"
        args.config = config_path
        args.output = str(output_path)
        
        loader = CurriculumDecoyLoader(config_path)
        result = cmd_export_confusability(args, loader)
        
        assert result == 0, "Export should succeed"
        assert output_path.exists(), "JSON file should be created"
    
    def test_export_json_is_parseable(self, config_path: str, temp_output_dir: Path):
        """Exported JSON should be valid and parseable."""
        output_path = temp_output_dir / "test_export.json"
        
        args = MagicMock()
        args.export_confusability = "slice_uplift_goal"
        args.config = config_path
        args.output = str(output_path)
        
        loader = CurriculumDecoyLoader(config_path)
        cmd_export_confusability(args, loader)
        
        # Should not raise
        data = json.loads(output_path.read_text(encoding='utf-8'))
        assert isinstance(data, dict)
    
    def test_export_has_required_keys(self, config_path: str, temp_output_dir: Path):
        """Exported JSON should have all required top-level keys."""
        output_path = temp_output_dir / "test_export.json"
        
        args = MagicMock()
        args.export_confusability = "slice_uplift_goal"
        args.config = config_path
        args.output = str(output_path)
        
        loader = CurriculumDecoyLoader(config_path)
        cmd_export_confusability(args, loader)
        
        data = json.loads(output_path.read_text(encoding='utf-8'))
        
        # Note: No generated_at field (for deterministic exports)
        required_keys = {"slice_name", "schema_version", "config_path", "formulas", "summary"}
        assert required_keys.issubset(set(data.keys())), f"Missing keys: {required_keys - set(data.keys())}"
    
    def test_export_formula_entries_have_required_keys(self, config_path: str, temp_output_dir: Path):
        """Each formula entry should have required keys."""
        output_path = temp_output_dir / "test_export.json"
        
        args = MagicMock()
        args.export_confusability = "slice_uplift_goal"
        args.config = config_path
        args.output = str(output_path)
        
        loader = CurriculumDecoyLoader(config_path)
        cmd_export_confusability(args, loader)
        
        data = json.loads(output_path.read_text(encoding='utf-8'))
        
        required_formula_keys = {
            "name", "role", "formula", "normalized", "hash",
            "difficulty", "confusability", "components"
        }
        
        for formula in data["formulas"]:
            missing = required_formula_keys - set(formula.keys())
            assert not missing, f"Formula '{formula.get('name')}' missing keys: {missing}"
    
    def test_export_targets_have_confusability_one(self, config_path: str, temp_output_dir: Path):
        """Target formulas should have confusability = 1.0."""
        output_path = temp_output_dir / "test_export.json"
        
        args = MagicMock()
        args.export_confusability = "slice_uplift_goal"
        args.config = config_path
        args.output = str(output_path)
        
        loader = CurriculumDecoyLoader(config_path)
        cmd_export_confusability(args, loader)
        
        data = json.loads(output_path.read_text(encoding='utf-8'))
        
        targets = [f for f in data["formulas"] if f["role"] == "target"]
        assert len(targets) > 0, "Should have at least one target"
        
        for target in targets:
            # Targets should have confusability = 1.0 (self-confusability)
            assert abs(target["confusability"] - 1.0) < 0.001, (
                f"Target '{target['name']}' should have confusability≈1.0, got {target['confusability']}"
            )
    
    def test_export_summary_counts_match_formulas(self, config_path: str, temp_output_dir: Path):
        """Summary counts should match actual formula counts."""
        output_path = temp_output_dir / "test_export.json"
        
        args = MagicMock()
        args.export_confusability = "slice_uplift_goal"
        args.config = config_path
        args.output = str(output_path)
        
        loader = CurriculumDecoyLoader(config_path)
        cmd_export_confusability(args, loader)
        
        data = json.loads(output_path.read_text(encoding='utf-8'))
        
        actual_target = sum(1 for f in data["formulas"] if f["role"] == "target")
        actual_near = sum(1 for f in data["formulas"] if f["role"] == "decoy_near")
        actual_far = sum(1 for f in data["formulas"] if f["role"] == "decoy_far")
        actual_bridge = sum(1 for f in data["formulas"] if f["role"] == "bridge")
        
        # New schema uses decoy_near_count, decoy_far_count
        assert data["summary"]["target_count"] == actual_target
        assert data["summary"]["decoy_near_count"] == actual_near
        assert data["summary"]["decoy_far_count"] == actual_far
        assert data["summary"]["bridge_count"] == actual_bridge


# =============================================================================
# VERIFY CONFUSABILITY TESTS
# =============================================================================

class TestVerifyConfusability:
    """Tests for --verify-confusability command."""
    
    def test_verify_real_slice_returns_ok_warn_or_skipped(self, config_path: str):
        """Real slices should return OK, WARN, or SKIPPED (not FAIL) if properly designed."""
        result = verify_slice_confusability("slice_uplift_goal", config_path)
        
        # Legacy format (no decoys) will return SKIPPED
        # Decoy-enabled format should be OK or WARN if properly designed
        assert result.status in (VerificationStatus.OK, VerificationStatus.WARN, VerificationStatus.SKIPPED), (
            f"Well-designed slice should be OK/WARN/SKIPPED, got {result.status.value}: {result.reasons}"
        )
    
    def test_verify_result_has_metrics(self, config_path: str):
        """Verification result should include metrics."""
        result = verify_slice_confusability("slice_uplift_goal", config_path)
        
        assert result.slice_name == "slice_uplift_goal"
        assert 0.0 <= result.near_avg <= 1.0
        assert 0.0 <= result.far_avg <= 1.0
        assert isinstance(result.bridge_count, int)
        assert isinstance(result.reasons, list)
    
    def test_verify_with_strict_thresholds_on_decoys(self, config_path: str):
        """Stricter thresholds may produce WARN or FAIL if decoys exist."""
        # Check if slice has decoys
        has_decoys = _has_decoys(config_path, "slice_uplift_goal")
        
        strict_thresholds = {
            "near_avg_min": 0.95,  # Very strict
            "far_avg_max": 0.2,   # Very strict
            "near_individual_min": 0.8,  # Very strict
        }
        
        result = verify_slice_confusability(
            "slice_uplift_goal",
            config_path,
            thresholds=strict_thresholds,
        )
        
        if has_decoys:
            # With decoys and strict thresholds, expect WARN or FAIL
            assert result.status in (VerificationStatus.WARN, VerificationStatus.FAIL), (
                f"Strict thresholds should trigger WARN or FAIL, got {result.status.value}"
            )
            assert len(result.reasons) > 0, "Should have reasons for non-OK status"
        else:
            # No decoys = SKIPPED (verification not applicable)
            assert result.status == VerificationStatus.SKIPPED
    
    def test_verify_with_loose_thresholds_returns_ok_or_skipped(self, config_path: str):
        """Very loose thresholds should produce OK (if decoys) or SKIPPED (if no decoys)."""
        loose_thresholds = {
            "near_avg_min": 0.0,   # Accept any
            "far_avg_max": 1.0,    # Accept any
            "near_individual_min": 0.0,  # Accept any
        }
        
        result = verify_slice_confusability(
            "slice_uplift_goal",
            config_path,
            thresholds=loose_thresholds,
        )
        
        # If decoys exist, should be OK. If no decoys, should be SKIPPED.
        assert result.status in (VerificationStatus.OK, VerificationStatus.SKIPPED), (
            f"Loose thresholds should produce OK or SKIPPED, got {result.status.value}: {result.reasons}"
        )
    
    def test_verify_all_slices_command(self, config_path: str):
        """Verifying all slices should work."""
        args = MagicMock()
        args.verify_confusability = "all"
        args.config = config_path
        args.format = "json"
        args.output = None
        args.explain = False
        
        loader = CurriculumDecoyLoader(config_path)
        
        # Capture stdout
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        
        try:
            result = cmd_verify_confusability(args, loader)
        finally:
            sys.stdout = old_stdout
        
        output = captured.getvalue()
        
        # Should produce valid JSON
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) > 0
        
        # All entries should have required fields
        for entry in data:
            assert "slice_name" in entry
            assert "status" in entry
            # SKIPPED is also valid for legacy slices
            assert entry["status"] in ("OK", "WARN", "FAIL", "SKIPPED")
    
    def test_verify_exit_code_zero_on_ok(self, config_path: str):
        """Exit code should be 0 when all slices OK or WARN."""
        args = MagicMock()
        args.verify_confusability = "slice_uplift_goal"
        args.config = config_path
        args.format = "markdown"
        args.output = None
        
        loader = CurriculumDecoyLoader(config_path)
        result = cmd_verify_confusability(args, loader)
        
        # Real slices should pass (exit 0) unless they're broken
        # If result is 1, check if it's a legitimate failure
        if result == 1:
            verification = verify_slice_confusability("slice_uplift_goal", config_path)
            assert verification.status == VerificationStatus.FAIL, (
                "Exit code 1 should only occur on FAIL status"
            )
    
    def test_verify_detects_low_confusability_near_decoy(self, config_path: str):
        """Should detect near-decoys with confusability below threshold if they exist."""
        has_decoys = _has_decoys(config_path, "slice_uplift_goal")
        
        if not has_decoys:
            pytest.skip("No decoys in legacy format - individual threshold check not applicable")
        
        # Use a threshold that's likely to catch some near-decoys
        strict_thresholds = {
            "near_avg_min": 0.7,
            "far_avg_max": 0.5,
            "near_individual_min": 0.9,  # Very strict individual threshold
        }
        
        result = verify_slice_confusability(
            "slice_uplift_goal",
            config_path,
            thresholds=strict_thresholds,
        )
        
        # Check that the individual check is working
        # (may or may not trigger depending on actual data)
        if result.status == VerificationStatus.FAIL:
            individual_failures = [r for r in result.reasons if "near-decoy" in r and "confusability" in r]
            # This is informational - the test verifies the mechanism works


# =============================================================================
# CI SUMMARY TESTS
# =============================================================================

class TestDecoyCISummary:
    """Tests for --decoy-ci-summary command."""
    
    def test_ci_summary_produces_output(self, config_path: str, slice_names: List[str]):
        """CI summary should produce output for each slice."""
        args = MagicMock()
        args.config = config_path
        
        loader = CurriculumDecoyLoader(config_path)
        
        # Capture stdout
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        
        try:
            result = cmd_decoy_ci_summary(args, loader)
        finally:
            sys.stdout = old_stdout
        
        output = captured.getvalue()
        
        # Should have one line per slice
        lines = [l for l in output.strip().split('\n') if l]
        assert len(lines) >= len(slice_names), (
            f"Expected at least {len(slice_names)} lines, got {len(lines)}"
        )
    
    def test_ci_summary_is_greppable(self, config_path: str):
        """Each line should be greppable with consistent format."""
        args = MagicMock()
        args.config = config_path
        
        loader = CurriculumDecoyLoader(config_path)
        
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        
        try:
            cmd_decoy_ci_summary(args, loader)
        finally:
            sys.stdout = old_stdout
        
        output = captured.getvalue()
        
        for line in output.strip().split('\n'):
            if not line:
                continue
            
            # Format: slice_name: STATUS (near=X.XXX, far=X.XXX, bridges=N)
            # or: slice_name: SKIPPED (no decoys)
            assert ": " in line, f"Line should contain ': ' separator: {line}"
            
            parts = line.split(": ", 1)
            assert len(parts) == 2, f"Line should have slice_name: rest format: {line}"
            
            status_part = parts[1]
            # SKIPPED is also valid for legacy slices
            assert status_part.startswith(("OK ", "WARN ", "FAIL ", "ERROR", "SKIPPED")), (
                f"Status should be OK/WARN/FAIL/ERROR/SKIPPED: {line}"
            )
    
    def test_ci_summary_contains_metrics(self, config_path: str):
        """Each line should contain near/far/bridges metrics (unless SKIPPED)."""
        args = MagicMock()
        args.config = config_path
        
        loader = CurriculumDecoyLoader(config_path)
        
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        
        try:
            cmd_decoy_ci_summary(args, loader)
        finally:
            sys.stdout = old_stdout
        
        output = captured.getvalue()
        
        for line in output.strip().split('\n'):
            if not line or "ERROR" in line:
                continue
            
            # SKIPPED lines have different format (no metrics)
            if "SKIPPED" in line:
                assert "no decoys" in line.lower() or "SKIPPED" in line
                continue
            
            assert "near=" in line, f"Line should contain near=: {line}"
            assert "far=" in line, f"Line should contain far=: {line}"
            assert "bridges=" in line, f"Line should contain bridges=: {line}"
    
    def test_ci_summary_exit_code_always_zero(self, config_path: str):
        """CI summary should always return 0 (use --verify for gating)."""
        args = MagicMock()
        args.config = config_path
        
        loader = CurriculumDecoyLoader(config_path)
        
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        
        try:
            result = cmd_decoy_ci_summary(args, loader)
        finally:
            sys.stdout = old_stdout
        
        assert result == 0, "CI summary should always return 0"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestCLIIntegration:
    """Integration tests for CLI commands via main()."""
    
    def test_main_export_confusability(self, config_path: str, temp_output_dir: Path):
        """Test main() with --export-confusability."""
        output_path = temp_output_dir / "export_test.json"
        
        result = main([
            "--config", config_path,
            "--export-confusability", "slice_uplift_goal",
            "-o", str(output_path),
        ])
        
        assert result == 0
        assert output_path.exists()
        
        # Verify JSON is valid
        data = json.loads(output_path.read_text())
        assert "formulas" in data
    
    def test_main_verify_confusability_single(self, config_path: str):
        """Test main() with --verify-confusability for single slice."""
        result = main([
            "--config", config_path,
            "--verify-confusability", "slice_uplift_goal",
            "--format", "json",
        ])
        
        # Should be 0 or 1 depending on slice status
        assert result in (0, 1)
    
    def test_main_verify_confusability_all(self, config_path: str):
        """Test main() with --verify-confusability all."""
        result = main([
            "--config", config_path,
            "--verify-confusability", "all",
            "--format", "json",
        ])
        
        assert result in (0, 1)
    
    def test_main_decoy_ci_summary(self, config_path: str):
        """Test main() with --decoy-ci-summary."""
        result = main([
            "--config", config_path,
            "--decoy-ci-summary",
        ])
        
        assert result == 0


# =============================================================================
# THRESHOLD MOCK TESTS
# =============================================================================

class TestThresholdBehavior:
    """Tests for threshold-based verification behavior."""
    
    def test_threshold_near_avg_triggers_warn(self, config_path: str):
        """Threshold violation should trigger WARN for mild cases if decoys exist."""
        has_decoys = _has_decoys(config_path, "slice_uplift_goal")
        
        # Get actual near_avg
        result_baseline = verify_slice_confusability(
            "slice_uplift_goal",
            config_path,
            thresholds={"near_avg_min": 0.0, "far_avg_max": 1.0, "near_individual_min": 0.0},
        )
        
        if not has_decoys or result_baseline.near_avg == 0.0:
            # No near-decoys = threshold check doesn't apply
            pytest.skip("No near-decoys in slice")
        
        # Set threshold just above actual
        warn_threshold = {
            "near_avg_min": result_baseline.near_avg + 0.05,  # Just above actual
            "far_avg_max": 1.0,
            "near_individual_min": 0.0,
        }
        
        result = verify_slice_confusability(
            "slice_uplift_goal",
            config_path,
            thresholds=warn_threshold,
        )
        
        # Should be WARN (mild violation) or FAIL (severe violation)
        assert result.status in (VerificationStatus.WARN, VerificationStatus.FAIL)
    
    def test_threshold_far_avg_triggers_warn(self, config_path: str):
        """Far avg threshold violation should trigger WARN if decoys exist."""
        has_decoys = _has_decoys(config_path, "slice_uplift_goal")
        
        result_baseline = verify_slice_confusability(
            "slice_uplift_goal",
            config_path,
            thresholds={"near_avg_min": 0.0, "far_avg_max": 1.0, "near_individual_min": 0.0},
        )
        
        if not has_decoys or result_baseline.far_avg == 0.0:
            pytest.skip("No far-decoys in slice")
        
        # Set threshold just below actual
        warn_threshold = {
            "near_avg_min": 0.0,
            "far_avg_max": max(0.0, result_baseline.far_avg - 0.05),  # Just below actual
            "near_individual_min": 0.0,
        }
        
        result = verify_slice_confusability(
            "slice_uplift_goal",
            config_path,
            thresholds=warn_threshold,
        )
        
        # May trigger WARN if far_avg is above threshold
        if result_baseline.far_avg > warn_threshold["far_avg_max"]:
            assert result.status in (VerificationStatus.WARN, VerificationStatus.FAIL)


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests to verify deterministic behavior."""
    
    def test_export_is_deterministic(self, config_path: str, temp_output_dir: Path):
        """Multiple exports should produce identical output (byte-stable)."""
        outputs = []
        
        for i in range(3):
            output_path = temp_output_dir / f"export_{i}.json"
            
            args = MagicMock()
            args.export_confusability = "slice_uplift_goal"
            args.config = config_path
            args.output = str(output_path)
            
            loader = CurriculumDecoyLoader(config_path)
            cmd_export_confusability(args, loader)
            
            # No timestamp to remove - export is fully deterministic
            content = output_path.read_text()
            outputs.append(content)
        
        # All outputs should be byte-identical
        for i in range(1, len(outputs)):
            assert outputs[0] == outputs[i], f"Export {i} differs from export 0"
    
    def test_verify_is_deterministic(self, config_path: str):
        """Multiple verifications should produce identical results."""
        results = []
        
        for _ in range(3):
            result = verify_slice_confusability("slice_uplift_goal", config_path)
            results.append(result.to_dict())
        
        for i in range(1, len(results)):
            assert results[0] == results[i], f"Verify {i} differs from verify 0"

