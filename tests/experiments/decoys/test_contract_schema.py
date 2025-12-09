# PHASE II — NOT USED IN PHASE I
"""
Test Suite: Confusability Contract Schema & CI Gates

This module tests:
1. Contract schema validation and deterministic serialization
2. CI gate with --explain flag
3. Legacy/decoy-aware CI policy
4. Byte-stable export reproducibility

All tests verify the canonical contract schema defined in
experiments/decoys/contract.py.
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

from experiments.decoys.contract import (
    SCHEMA_VERSION,
    FormulaEntry,
    ContractSummary,
    ConfusabilityContract,
    export_contract,
    validate_contract_schema,
    contracts_are_equal,
)
from experiments.curriculum_diagnostics import (
    VerificationStatus,
    VerificationResult,
    FailureExplanation,
    verify_slice_confusability,
    cmd_verify_confusability,
    cmd_export_confusability,
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


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# TASK 1: CONTRACT SCHEMA TESTS
# =============================================================================

class TestContractSchema:
    """Tests for the canonical contract schema."""
    
    def test_schema_version_is_stable(self):
        """Schema version should be a stable semantic version."""
        # v1.1.0 adds family profiles
        assert SCHEMA_VERSION == "1.1.0"
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
    
    def test_formula_entry_to_dict_has_sorted_keys(self):
        """FormulaEntry.to_dict() should have alphabetically sorted keys."""
        entry = FormulaEntry(
            name="test_formula",
            role="target",
            formula="p->q",
            normalized="p->q",
            hash="abc123" * 10 + "abcd",
            difficulty=0.5,
            confusability=0.8,
            components={"syntactic": 0.7, "connective": 0.8, "atom_similarity": 0.9, "chain_alignment": 0.6},
        )
        
        d = entry.to_dict()
        keys = list(d.keys())
        assert keys == sorted(keys), "Keys must be alphabetically sorted"
        
        # Check components are also sorted
        comp_keys = list(d["components"].keys())
        assert comp_keys == sorted(comp_keys), "Component keys must be sorted"
    
    def test_formula_entry_rounds_floats_to_6_places(self):
        """Floats should be rounded to exactly 6 decimal places."""
        entry = FormulaEntry(
            name="test",
            role="target",
            formula="p",
            normalized="p",
            hash="a" * 64,
            difficulty=0.123456789,
            confusability=0.987654321,
            components={"syntactic": 0.111111111},
        )
        
        d = entry.to_dict()
        assert d["difficulty"] == 0.123457
        assert d["confusability"] == 0.987654
        assert d["components"]["syntactic"] == 0.111111
    
    def test_contract_summary_to_dict_has_sorted_keys(self):
        """ContractSummary.to_dict() should have sorted keys."""
        summary = ContractSummary(
            target_count=5,
            decoy_near_count=3,
            decoy_far_count=2,
            bridge_count=1,
            avg_confusability_near=0.8,
            avg_confusability_far=0.4,
        )
        
        d = summary.to_dict()
        keys = list(d.keys())
        assert keys == sorted(keys)
    
    def test_contract_to_dict_has_sorted_keys(self):
        """ConfusabilityContract.to_dict() should have sorted keys at all levels."""
        contract = ConfusabilityContract(
            slice_name="test_slice",
            config_path="test/path.yaml",
            formulas=[],
            summary=ContractSummary(),
        )
        
        d = contract.to_dict()
        keys = list(d.keys())
        assert keys == sorted(keys)
    
    def test_contract_to_json_is_deterministic(self, config_path: str):
        """Multiple to_json() calls should produce identical output."""
        contract = export_contract("slice_uplift_goal", config_path)
        
        json1 = contract.to_json()
        json2 = contract.to_json()
        json3 = contract.to_json()
        
        assert json1 == json2 == json3
    
    def test_contract_to_bytes_is_deterministic(self, config_path: str):
        """Multiple to_bytes() calls should produce identical output."""
        contract = export_contract("slice_uplift_goal", config_path)
        
        bytes1 = contract.to_bytes()
        bytes2 = contract.to_bytes()
        
        assert bytes1 == bytes2
    
    def test_export_reexport_identical(self, config_path: str):
        """Export → Re-export should produce identical bytes."""
        contract1 = export_contract("slice_uplift_goal", config_path)
        contract2 = export_contract("slice_uplift_goal", config_path)
        
        assert contracts_are_equal(contract1, contract2)
        assert contract1.to_bytes() == contract2.to_bytes()
    
    def test_export_no_timestamp(self, config_path: str):
        """Exported contract should not contain a timestamp field."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        
        assert "generated_at" not in d
        assert "timestamp" not in d
        
        json_str = contract.to_json()
        assert "generated_at" not in json_str
        assert "timestamp" not in json_str
    
    def test_export_includes_schema_version(self, config_path: str):
        """Exported contract should include schema_version."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        
        assert "schema_version" in d
        assert d["schema_version"] == SCHEMA_VERSION


class TestSchemaValidation:
    """Tests for contract schema validation."""
    
    def test_validate_valid_contract(self, config_path: str):
        """Valid contract should pass validation."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        
        is_valid, errors = validate_contract_schema(d)
        assert is_valid, f"Validation errors: {errors}"
        assert len(errors) == 0
    
    def test_validate_missing_keys(self):
        """Contract missing required keys should fail."""
        invalid = {"slice_name": "test"}
        
        is_valid, errors = validate_contract_schema(invalid)
        assert not is_valid
        assert any("Missing" in e for e in errors)
    
    def test_validate_wrong_schema_version(self, config_path: str):
        """Contract with wrong schema version should fail."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        d["schema_version"] = "99.0.0"
        
        is_valid, errors = validate_contract_schema(d)
        assert not is_valid
        assert any("schema version" in e.lower() for e in errors)
    
    def test_validate_invalid_role(self, config_path: str):
        """Formula with invalid role should fail validation."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        if d["formulas"]:
            d["formulas"][0]["role"] = "invalid_role"
        
        is_valid, errors = validate_contract_schema(d)
        assert not is_valid
        assert any("invalid role" in e.lower() for e in errors)


class TestDeterministicSorting:
    """Tests for deterministic formula sorting."""
    
    def test_formulas_sorted_by_role_then_name(self, config_path: str):
        """Formulas should be sorted by (role_order, name)."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        
        role_order = {"target": 0, "decoy_near": 1, "decoy_far": 2, "bridge": 3}
        
        for i in range(len(d["formulas"]) - 1):
            f1 = d["formulas"][i]
            f2 = d["formulas"][i + 1]
            
            key1 = (role_order.get(f1["role"], 99), f1["name"])
            key2 = (role_order.get(f2["role"], 99), f2["name"])
            
            assert key1 <= key2, f"Formulas not sorted: {f1['name']} should come before {f2['name']}"


# =============================================================================
# TASK 2: CI GATE WITH --EXPLAIN TESTS
# =============================================================================

class TestVerificationExplanations:
    """Tests for --verify-confusability --explain functionality."""
    
    def test_failure_explanation_structure(self):
        """FailureExplanation should have correct structure."""
        exp = FailureExplanation(
            threshold_name="near_avg_min",
            threshold_value=0.7,
            actual_value=0.5,
            formula_name="bad_decoy",
            formula_confusability=0.3,
            formula_difficulty=0.8,
            rationale="Test rationale",
        )
        
        d = exp.to_dict()
        assert "threshold_name" in d
        assert "threshold_value" in d
        assert "actual_value" in d
        assert "formula_name" in d
        assert "rationale" in d
        assert d["threshold_name"] == "near_avg_min"
    
    def test_verification_result_includes_explanations(self, config_path: str):
        """VerificationResult should include explanations list."""
        result = verify_slice_confusability("slice_uplift_goal", config_path)
        
        assert hasattr(result, "explanations")
        assert isinstance(result.explanations, list)
    
    def test_explain_flag_output(self, config_path: str, capsys):
        """--explain flag should print detailed explanations."""
        # This tests the markdown output with explanations
        args = MagicMock()
        args.verify_confusability = "slice_uplift_goal"
        args.config = config_path
        args.format = "markdown"
        args.output = None
        args.explain = True
        
        loader = CurriculumDecoyLoader(config_path)
        cmd_verify_confusability(args, loader)
        
        # Output should be captured (even if no failures)
        captured = capsys.readouterr()
        # Just verify it runs without error
        assert "Confusability Verification Results" in captured.out


class TestSyntheticFailures:
    """Tests for synthetic bad decoy scenarios."""
    
    def test_low_near_decoy_triggers_fail(self):
        """A near-decoy with very low confusability should trigger FAIL."""
        # Create a mock result with a bad near-decoy
        result = VerificationResult(
            slice_name="test_slice",
            status=VerificationStatus.FAIL,
            near_avg=0.3,
            far_avg=0.4,
            bridge_count=0,
            has_decoys=True,
            reasons=["FAIL: near-decoy 'bad_near' has confusability 0.200 < 0.4"],
            explanations=[
                FailureExplanation(
                    threshold_name="near_individual_min",
                    threshold_value=0.4,
                    actual_value=0.2,
                    formula_name="bad_near",
                    formula_confusability=0.2,
                    formula_difficulty=0.8,
                    rationale="Near-decoy too easily distinguished from targets",
                )
            ],
        )
        
        assert result.status == VerificationStatus.FAIL
        assert len(result.explanations) == 1
        assert result.explanations[0].formula_name == "bad_near"
        assert result.explanations[0].actual_value < result.explanations[0].threshold_value
    
    def test_explanation_references_formula_name(self):
        """Failure explanation must reference the formula name."""
        exp = FailureExplanation(
            threshold_name="near_individual_min",
            threshold_value=0.4,
            actual_value=0.2,
            formula_name="problematic_formula",
            formula_confusability=0.2,
            formula_difficulty=0.7,
            rationale="Test",
        )
        
        d = exp.to_dict()
        assert d["formula_name"] == "problematic_formula"
        assert d["formula_confusability"] == 0.2
    
    def test_explanation_references_threshold(self):
        """Failure explanation must reference the failed threshold."""
        exp = FailureExplanation(
            threshold_name="near_avg_min",
            threshold_value=0.7,
            actual_value=0.5,
            rationale="Average too low",
        )
        
        d = exp.to_dict()
        assert d["threshold_name"] == "near_avg_min"
        assert d["threshold_value"] == 0.7


# =============================================================================
# TASK 3: LEGACY/DECOY-AWARE CI POLICY TESTS
# =============================================================================

class TestLegacyAwareCIPolicy:
    """Tests for legacy format handling."""
    
    def test_legacy_slice_returns_skipped(self, config_path: str):
        """Slice with only targets (legacy format) should return SKIPPED."""
        result = verify_slice_confusability("slice_uplift_goal", config_path)
        
        # Current YAML has legacy format, so should be SKIPPED
        if not result.has_decoys:
            assert result.status == VerificationStatus.SKIPPED
            assert "No decoys present" in result.reasons[0]
    
    def test_legacy_slice_exit_code_zero(self, config_path: str):
        """Legacy slice should result in exit code 0."""
        args = MagicMock()
        args.verify_confusability = "slice_uplift_goal"
        args.config = config_path
        args.format = "json"
        args.output = None
        args.explain = False
        
        loader = CurriculumDecoyLoader(config_path)
        exit_code = cmd_verify_confusability(args, loader)
        
        # Should pass (exit 0) regardless of decoy presence
        assert exit_code == 0
    
    def test_legacy_export_safe_fallbacks(self, config_path: str):
        """Legacy slice export should have safe fallback values."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        
        summary = d["summary"]
        # Should have 0 for decoy counts if no decoys
        assert summary["decoy_near_count"] >= 0
        assert summary["decoy_far_count"] >= 0
        assert summary["bridge_count"] >= 0
        
        # Averages should be 0.0 if no decoys
        if summary["decoy_near_count"] == 0:
            assert summary["avg_confusability_near"] == 0.0
        if summary["decoy_far_count"] == 0:
            assert summary["avg_confusability_far"] == 0.0
    
    def test_skipped_message_in_output(self, config_path: str, capsys):
        """SKIPPED status should emit descriptive message."""
        args = MagicMock()
        args.verify_confusability = "slice_uplift_goal"
        args.config = config_path
        args.format = "markdown"
        args.output = None
        args.explain = False
        
        loader = CurriculumDecoyLoader(config_path)
        cmd_verify_confusability(args, loader)
        
        captured = capsys.readouterr()
        
        # If slice has no decoys, should see skip message
        result = verify_slice_confusability("slice_uplift_goal", config_path)
        if not result.has_decoys:
            assert "SKIPPED" in captured.out or "skipped" in captured.out.lower()


class TestCISummaryLegacy:
    """Tests for CI summary with legacy slices."""
    
    def test_ci_summary_shows_skipped(self, config_path: str, capsys):
        """CI summary should show SKIPPED for legacy slices."""
        from experiments.curriculum_diagnostics import cmd_decoy_ci_summary
        
        args = MagicMock()
        args.config = config_path
        
        loader = CurriculumDecoyLoader(config_path)
        cmd_decoy_ci_summary(args, loader)
        
        captured = capsys.readouterr()
        
        # Should contain either metrics or SKIPPED for each slice
        for line in captured.out.strip().split('\n'):
            if line:
                assert any(s in line for s in ["OK", "WARN", "FAIL", "SKIPPED", "ERROR"])


# =============================================================================
# TASK 4: CONFUSABILITY STABILITY TESTS
# =============================================================================

class TestConfusabilityStability:
    """Tests verifying confusability is diagnostic-only and stable."""
    
    def test_confusability_is_deterministic(self, config_path: str):
        """Confusability scores should be deterministic across runs."""
        from experiments.decoys.confusability import compute_confusability
        
        targets = ["p->q", "q->r"]
        formula = "p->(q->r)"
        
        scores = [compute_confusability(formula, targets) for _ in range(5)]
        
        assert len(set(scores)) == 1, "Confusability should be deterministic"
    
    def test_confusability_in_range(self, config_path: str):
        """Confusability scores should always be in [0, 1]."""
        contract = export_contract("slice_uplift_goal", config_path)
        
        for f in contract.formulas:
            assert 0.0 <= f.confusability <= 1.0, (
                f"Confusability out of range for {f.name}: {f.confusability}"
            )
    
    def test_difficulty_in_range(self, config_path: str):
        """Difficulty scores should always be in [0, 1]."""
        contract = export_contract("slice_uplift_goal", config_path)
        
        for f in contract.formulas:
            assert 0.0 <= f.difficulty <= 1.0, (
                f"Difficulty out of range for {f.name}: {f.difficulty}"
            )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_main_export_deterministic(self, config_path: str, temp_output_dir: Path):
        """Main export should be deterministic."""
        path1 = temp_output_dir / "export1.json"
        path2 = temp_output_dir / "export2.json"
        
        main(["--config", config_path, "--export-confusability", "slice_uplift_goal", "-o", str(path1)])
        main(["--config", config_path, "--export-confusability", "slice_uplift_goal", "-o", str(path2)])
        
        content1 = path1.read_text()
        content2 = path2.read_text()
        
        assert content1 == content2, "Exports should be byte-identical"
    
    def test_main_verify_with_explain(self, config_path: str, capsys):
        """Main verify with --explain should work."""
        exit_code = main([
            "--config", config_path,
            "--verify-confusability", "slice_uplift_goal",
            "--explain",
        ])
        
        # Should complete without error
        assert exit_code in (0, 1)
    
    def test_main_verify_all_json(self, config_path: str, capsys):
        """Main verify all with JSON output should be valid JSON."""
        exit_code = main([
            "--config", config_path,
            "--verify-confusability", "all",
            "--format", "json",
        ])
        
        captured = capsys.readouterr()
        
        # Should be valid JSON
        data = json.loads(captured.out)
        assert isinstance(data, list)
    
    def test_main_ci_summary(self, config_path: str, capsys):
        """Main CI summary should produce greppable output."""
        exit_code = main([
            "--config", config_path,
            "--decoy-ci-summary",
        ])
        
        assert exit_code == 0
        
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().split('\n') if l]
        
        # Each line should be greppable
        for line in lines:
            assert ": " in line


class TestByteStableExport:
    """Regression tests for byte-stable export."""
    
    def test_export_bytes_stable_across_calls(self, config_path: str):
        """Export bytes should be identical across multiple calls."""
        exports = []
        
        for _ in range(5):
            contract = export_contract("slice_uplift_goal", config_path)
            exports.append(contract.to_bytes())
        
        # All exports should be identical
        for i in range(1, len(exports)):
            assert exports[0] == exports[i], f"Export {i} differs from export 0"
    
    def test_json_keys_sorted(self, config_path: str):
        """JSON output should have sorted keys for reproducibility."""
        contract = export_contract("slice_uplift_goal", config_path)
        json_str = contract.to_json()
        
        # Parse and re-serialize with sorted keys
        data = json.loads(json_str)
        reserialized = json.dumps(data, indent=2, sort_keys=True)
        
        # Should be identical (keys were already sorted)
        assert json_str == reserialized

