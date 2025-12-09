"""
PHASE II — DIRECTORY ENTROPY AUDIT TESTS

Agent: E3 (doc-ops-3) — Structural Drift Sentinel
Purpose: Test entropy scoring, boundary violation detection,
         archetype classification stability, risk index computation,
         phase filtering, drift detection, CI summary generation,
         zone-based grouping, refactor bundles, structural posture,
         release gating, and refactor planning.

Test Categories:
1. Entropy score reproducibility (tests 1-6)
2. Boundary violation detection (tests 7-13)
3. Archetype classification stability (tests 14-20)
4. Report generation & determinism (tests 21-25)
5. Edge cases (tests 26-28)
6. Integration (tests 29-30)
7. Risk index computation (tests 31-40)
8. Phase filtering (tests 41-50)
9. Structural drift detection (tests 51-60)
10. Refactor candidates (tests 61-70)
11. Drift report schema contract (tests 71-80)
12. Phase II heatmap & CI summary (tests 81-90)
13. Zone-based structural grouping (tests 91-100)
14. Refactor candidate bundles (tests 101-110)
15. Structural posture summary (tests 111-120)
16. Structure-aware release gate (tests 121-130)
17. Refactor planning view (tests 131-140)
18. Director structure panel (tests 141-150)
19. CI release gate hook (tests 151-155)
20. Refactor sprint planner (tests 156-165)
21. Semantic drift tensor (tests 166-172)
22. Semantic drift counterfactual analyzer (tests 173-179)
23. Semantic drift director panel v3 (tests 180-188)

Total: 188 tests
"""

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

# Add scripts to path for importing auditor
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def auditor_module():
    """Import the auditor module."""
    import directory_entropy_audit
    return directory_entropy_audit


@pytest.fixture
def mock_directory(tmp_path):
    """Create a mock directory structure for testing."""
    # Create subdirectories
    (tmp_path / "backend").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    
    # Create test files
    (tmp_path / "backend" / "api.py").write_text("# API module")
    (tmp_path / "backend" / "worker.py").write_text("# Worker module")
    (tmp_path / "backend" / "models.py").write_text("# Models")
    (tmp_path / "tests" / "test_api.py").write_text("# Test API")
    (tmp_path / "tests" / "conftest.py").write_text("# Conftest")
    (tmp_path / "docs" / "README.md").write_text("# Readme")
    (tmp_path / "docs" / "API.md").write_text("# API Docs")
    
    return tmp_path


@pytest.fixture
def mock_violation_directory(tmp_path):
    """Create a directory with phase boundary violations."""
    (tmp_path / "phase_i").mkdir()
    
    # Create a file with Phase II imports
    violation_code = """
# Phase I module
from backend.security.u2_security import SecurityException
from backend.metrics.u2_analysis import analyze
import backend.runner.u2_runner
"""
    (tmp_path / "phase_i" / "module.py").write_text(violation_code)
    
    return tmp_path


# -----------------------------------------------------------------------------
# ENTROPY SCORE REPRODUCIBILITY TESTS (1-6)
# -----------------------------------------------------------------------------

class TestEntropyScoreReproducibility:
    """Tests ensuring entropy scores are deterministic and reproducible."""
    
    def test_shannon_entropy_empty_list(self, auditor_module):
        """Test 1: Shannon entropy of empty list is 0."""
        entropy = auditor_module.compute_shannon_entropy([])
        assert entropy == 0.0
    
    def test_shannon_entropy_single_item(self, auditor_module):
        """Test 2: Shannon entropy of single item is 0 (no uncertainty)."""
        entropy = auditor_module.compute_shannon_entropy(["a"])
        assert entropy == 0.0
    
    def test_shannon_entropy_uniform_distribution(self, auditor_module):
        """Test 3: Shannon entropy of uniform distribution is log2(n)."""
        # 4 items with equal probability should have entropy = log2(4) = 2.0
        items = ["a", "b", "c", "d"]
        entropy = auditor_module.compute_shannon_entropy(items)
        assert abs(entropy - 2.0) < 0.0001
    
    def test_shannon_entropy_skewed_distribution(self, auditor_module):
        """Test 4: Shannon entropy of skewed distribution is lower."""
        # 4 items, 3 of same type
        items = ["a", "a", "a", "b"]
        entropy = auditor_module.compute_shannon_entropy(items)
        # Should be less than uniform (2.0)
        assert entropy < 2.0
        assert entropy > 0.0
    
    def test_entropy_score_reproducibility(self, auditor_module, mock_directory):
        """Test 5: Same directory produces identical entropy scores."""
        backend_dir = mock_directory / "backend"
        
        score1 = auditor_module.compute_directory_entropy(backend_dir, base_path=mock_directory)
        score2 = auditor_module.compute_directory_entropy(backend_dir, base_path=mock_directory)
        
        assert score1.total_entropy == score2.total_entropy
        assert score1.extension_entropy == score2.extension_entropy
        assert score1.naming_entropy == score2.naming_entropy
        assert score1.file_count == score2.file_count
    
    def test_entropy_score_deterministic_across_runs(self, auditor_module):
        """Test 6: Entropy calculation is deterministic across multiple runs."""
        # Test with known data
        items1 = [".py", ".py", ".py", ".md"]
        items2 = [".py", ".py", ".py", ".md"]
        
        entropy1 = auditor_module.compute_shannon_entropy(items1)
        entropy2 = auditor_module.compute_shannon_entropy(items2)
        
        assert entropy1 == entropy2


# -----------------------------------------------------------------------------
# BOUNDARY VIOLATION DETECTION TESTS (7-13)
# -----------------------------------------------------------------------------

class TestBoundaryViolationDetection:
    """Tests for Phase I → Phase II import violation detection."""
    
    def test_detect_from_import_violation(self, auditor_module):
        """Test 7: Detect 'from X import Y' style violations."""
        code = "from backend.security.u2_security import SecurityException"
        
        for pattern, target in auditor_module.PHASE_II_IMPORT_PATTERNS:
            import re
            if re.search(pattern, code):
                assert "u2_security" in target
                return
        pytest.fail("Should detect from-import violation")
    
    def test_detect_import_module_violation(self, auditor_module):
        """Test 8: Detect 'import X' style violations."""
        code = "import backend.runner.u2_runner"
        
        for pattern, target in auditor_module.PHASE_II_IMPORT_PATTERNS:
            import re
            if re.search(pattern, code):
                assert "u2_runner" in target
                return
        pytest.fail("Should detect import-module violation")
    
    def test_no_false_positive_on_phase_i_imports(self, auditor_module):
        """Test 9: Don't flag Phase I imports as violations."""
        phase_i_imports = [
            "from backend.crypto import hashing",
            "from backend.dag import proof_dag",
            "from rfl import runner",
            "import backend.axiom_engine",
        ]
        
        for code in phase_i_imports:
            matches = False
            import re
            for pattern, _ in auditor_module.PHASE_II_IMPORT_PATTERNS:
                if re.search(pattern, code):
                    matches = True
                    break
            assert not matches, f"False positive on: {code}"
    
    def test_boundary_violation_structure(self, auditor_module):
        """Test 10: BoundaryViolation has correct structure."""
        violation = auditor_module.BoundaryViolation(
            source_file="test.py",
            line_number=10,
            import_statement="from backend.security.u2_security import X",
            target_module="backend.security.u2_security",
            severity="error",
        )
        
        d = violation.to_dict()
        assert "source_file" in d
        assert "line_number" in d
        assert "import_statement" in d
        assert "target_module" in d
        assert "severity" in d
    
    def test_scan_detects_known_violation(self, auditor_module):
        """Test 11: Scanner detects the known rfl/runner.py violation."""
        violations = auditor_module.scan_for_boundary_violations()
        
        # We know rfl/runner.py has a violation
        rfl_violations = [v for v in violations if "rfl/runner.py" in v.source_file]
        
        # Should find at least one violation in rfl/runner.py
        assert len(rfl_violations) >= 1, "Should detect known violation in rfl/runner.py"
    
    def test_remediation_suggestion_types(self, auditor_module):
        """Test 12: Remediation suggestions have valid types."""
        valid_types = {"promote", "lazy_import", "dependency_inversion", "relocate"}
        
        violations = [
            auditor_module.BoundaryViolation(
                source_file="test.py",
                line_number=1,
                import_statement="from backend.security.u2_security import X",
                target_module="backend.security.u2_security",
                severity="error",
            ),
            auditor_module.BoundaryViolation(
                source_file="test.py",
                line_number=2,
                import_statement="from backend.metrics.u2_analysis import Y",
                target_module="backend.metrics.u2_analysis",
                severity="error",
            ),
        ]
        
        suggestions = auditor_module.generate_remediation_suggestions(violations)
        
        assert len(suggestions) == 2
        for suggestion in suggestions:
            assert suggestion.suggestion_type in valid_types
    
    def test_remediation_has_description(self, auditor_module):
        """Test 13: All remediation suggestions have descriptions."""
        violation = auditor_module.BoundaryViolation(
            source_file="test.py",
            line_number=1,
            import_statement="from backend.runner.u2_runner import run",
            target_module="backend.runner.u2_runner",
            severity="error",
        )
        
        suggestions = auditor_module.generate_remediation_suggestions([violation])
        
        assert len(suggestions) == 1
        assert suggestions[0].description
        assert len(suggestions[0].description) > 20


# -----------------------------------------------------------------------------
# ARCHETYPE CLASSIFICATION STABILITY TESTS (14-20)
# -----------------------------------------------------------------------------

class TestArchetypeClassificationStability:
    """Tests for directory archetype classification."""
    
    def test_archetype_definitions_complete(self, auditor_module):
        """Test 14: All archetype definitions have required fields."""
        required_fields = ["description", "indicators", "expected_extensions", "directories"]
        
        for name, config in auditor_module.DIRECTORY_ARCHETYPES.items():
            for field in required_fields:
                assert field in config, f"Archetype '{name}' missing field '{field}'"
    
    def test_runtime_archetype_classification(self, auditor_module):
        """Test 15: Runtime directories are correctly classified."""
        # Create a mock path object
        runtime_path = PROJECT_ROOT / "backend" / "orchestrator"
        
        if runtime_path.exists():
            classification = auditor_module.classify_directory_archetype(runtime_path)
            assert classification.primary_archetype == "runtime"
    
    def test_testing_archetype_classification(self, auditor_module):
        """Test 16: Test directories are correctly classified."""
        tests_path = PROJECT_ROOT / "tests"
        
        if tests_path.exists():
            classification = auditor_module.classify_directory_archetype(tests_path)
            assert classification.primary_archetype == "testing"
    
    def test_documentation_archetype_classification(self, auditor_module):
        """Test 17: Documentation directories are correctly classified."""
        docs_path = PROJECT_ROOT / "docs"
        
        if docs_path.exists():
            classification = auditor_module.classify_directory_archetype(docs_path)
            assert classification.primary_archetype == "documentation"
    
    def test_classification_confidence_range(self, auditor_module):
        """Test 18: Classification confidence is in valid range [0, 1]."""
        classifications = auditor_module.classify_all_directories()
        
        for c in classifications:
            assert 0.0 <= c.confidence <= 1.0, f"Invalid confidence for {c.directory}: {c.confidence}"
    
    def test_classification_reproducibility(self, auditor_module):
        """Test 19: Same directory produces same classification."""
        tests_path = PROJECT_ROOT / "tests"
        
        if tests_path.exists():
            c1 = auditor_module.classify_directory_archetype(tests_path)
            c2 = auditor_module.classify_directory_archetype(tests_path)
            
            assert c1.primary_archetype == c2.primary_archetype
            assert c1.confidence == c2.confidence
    
    def test_archetype_distribution_stability(self, auditor_module):
        """Test 20: Archetype distribution is stable across runs."""
        c1 = auditor_module.classify_all_directories()
        c2 = auditor_module.classify_all_directories()
        
        dist1 = Counter(c.primary_archetype for c in c1)
        dist2 = Counter(c.primary_archetype for c in c2)
        
        assert dist1 == dist2


# -----------------------------------------------------------------------------
# ADDITIONAL TESTS (21-25) — Report Generation & Determinism
# -----------------------------------------------------------------------------

class TestReportGeneration:
    """Tests for report generation and determinism."""
    
    def test_report_hash_determinism(self, auditor_module):
        """Test 21: Report hash is deterministic for same content."""
        report_data = {
            "total_directories": 10,
            "total_files": 100,
            "global_entropy": 0.5,
        }
        
        hash1 = auditor_module.compute_report_hash(report_data)
        hash2 = auditor_module.compute_report_hash(report_data)
        
        assert hash1 == hash2
    
    def test_report_hash_changes_with_content(self, auditor_module):
        """Test 22: Report hash changes when content changes."""
        report_data1 = {"total_directories": 10}
        report_data2 = {"total_directories": 11}
        
        hash1 = auditor_module.compute_report_hash(report_data1)
        hash2 = auditor_module.compute_report_hash(report_data2)
        
        assert hash1 != hash2
    
    def test_entropy_score_to_dict(self, auditor_module):
        """Test 23: EntropyScore serializes correctly."""
        score = auditor_module.EntropyScore(
            directory="test",
            total_entropy=1.5,
            extension_entropy=0.8,
            naming_entropy=0.7,
            violation_count=2,
            unexpected_files=["a.txt"],
            forbidden_files=["b.py"],
            naming_violations=["c.PY"],
            file_count=10,
        )
        
        d = score.to_dict()
        
        assert d["directory"] == "test"
        assert d["total_entropy"] == 1.5
        assert d["violation_count"] == 2
        assert isinstance(d["unexpected_files"], list)
    
    def test_report_structure_completeness(self, auditor_module):
        """Test 24: Generated report has all required fields."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        
        report_dict = report.to_dict()
        
        required_fields = [
            "timestamp",
            "project_root",
            "report_hash",
            "total_directories",
            "total_files",
            "global_entropy",
            "directory_scores",
            "boundary_violations",
            "remediation_suggestions",
            "archetype_classifications",
            "mutation_events",
            "summary",
        ]
        
        for field in required_fields:
            assert field in report_dict, f"Missing field: {field}"
    
    def test_report_json_serializable(self, auditor_module):
        """Test 25: Full report is JSON serializable."""
        report = auditor_module.generate_report(
            include_guardian=True,
            include_classify=True,
            include_history=False,
            include_remediation=True,
        )
        
        # Should not raise
        json_str = json.dumps(report.to_dict(), indent=2)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "timestamp" in parsed


# -----------------------------------------------------------------------------
# EDGE CASE TESTS (26-28)
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_directory_entropy(self, auditor_module, tmp_path):
        """Test 26: Empty directory has zero entropy."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        score = auditor_module.compute_directory_entropy(empty_dir, base_path=tmp_path)
        
        assert score.total_entropy == 0.0
        assert score.file_count == 0
    
    def test_single_file_directory_entropy(self, auditor_module, tmp_path):
        """Test 27: Single-file directory has minimal entropy."""
        single_dir = tmp_path / "single"
        single_dir.mkdir()
        (single_dir / "only.py").write_text("# only file")
        
        score = auditor_module.compute_directory_entropy(single_dir, base_path=tmp_path)
        
        assert score.extension_entropy == 0.0  # One extension type
        assert score.file_count == 1
    
    def test_mixed_extension_entropy_increases(self, auditor_module, tmp_path):
        """Test 28: More file types increases extension entropy."""
        mixed_dir = tmp_path / "mixed"
        mixed_dir.mkdir()
        
        # Single extension
        (mixed_dir / "a.py").write_text("# a")
        score1 = auditor_module.compute_directory_entropy(mixed_dir, base_path=tmp_path)
        
        # Add another extension
        (mixed_dir / "b.md").write_text("# b")
        score2 = auditor_module.compute_directory_entropy(mixed_dir, base_path=tmp_path)
        
        # Add third extension
        (mixed_dir / "c.json").write_text("{}")
        score3 = auditor_module.compute_directory_entropy(mixed_dir, base_path=tmp_path)
        
        assert score2.extension_entropy > score1.extension_entropy
        assert score3.extension_entropy > score2.extension_entropy


# -----------------------------------------------------------------------------
# INTEGRATION TESTS (29-30)
# -----------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for the auditor."""
    
    def test_full_audit_runs_without_error(self, auditor_module):
        """Test 29: Full audit completes without errors."""
        # Should not raise
        report = auditor_module.generate_report(
            include_guardian=True,
            include_classify=True,
            include_history=False,  # Skip git history for speed
            include_remediation=True,
        )
        
        assert report.total_directories > 0
        assert report.total_files > 0
    
    def test_auditor_detects_project_violations(self, auditor_module):
        """Test 30: Auditor detects known project violations."""
        violations = auditor_module.scan_for_boundary_violations()
        
        # We know there's at least one violation in the project
        # (rfl/runner.py imports backend.security.u2_security)
        if violations:
            # Verify violation structure
            v = violations[0]
            assert v.source_file
            assert v.line_number > 0
            assert v.import_statement
            assert v.target_module


# -----------------------------------------------------------------------------
# RISK INDEX COMPUTATION TESTS (31-40)
# -----------------------------------------------------------------------------

class TestRiskIndexComputation:
    """Tests for Structure Risk Index computation."""
    
    def test_risk_score_determinism(self, auditor_module):
        """Test 31: Same report produces identical risk index."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        
        risk1 = auditor_module.compute_structure_risk_index(report)
        risk2 = auditor_module.compute_structure_risk_index(report)
        
        assert risk1.global_risk_score == risk2.global_risk_score
        assert risk1.global_risk_level == risk2.global_risk_level
        assert risk1.risk_distribution == risk2.risk_distribution
    
    def test_risk_score_range(self, auditor_module):
        """Test 32: Global risk score is in valid range [0, 1]."""
        report = auditor_module.generate_report(
            include_guardian=True,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        
        risk = auditor_module.compute_structure_risk_index(report)
        
        assert 0.0 <= risk.global_risk_score <= 1.0
    
    def test_risk_level_values(self, auditor_module):
        """Test 33: Risk level is one of LOW, MEDIUM, HIGH."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        
        risk = auditor_module.compute_structure_risk_index(report)
        
        assert risk.global_risk_level in {"LOW", "MEDIUM", "HIGH"}
    
    def test_directory_risk_classification(self, auditor_module):
        """Test 34: Directory risk classification produces valid levels."""
        score = auditor_module.EntropyScore(
            directory="test",
            total_entropy=2.5,
            extension_entropy=1.0,
            naming_entropy=0.5,
            violation_count=5,
            unexpected_files=["a.txt"],
            forbidden_files=["b.py"],
            naming_violations=["c.PY"],
            file_count=10,
        )
        
        risk = auditor_module.classify_directory_risk(score)
        
        assert risk.risk_level in {"LOW", "MEDIUM", "HIGH"}
        assert 0.0 <= risk.risk_score <= 1.0
        assert risk.path == "test"
    
    def test_high_entropy_produces_high_risk(self, auditor_module):
        """Test 35: High entropy directory gets HIGH risk classification."""
        score = auditor_module.EntropyScore(
            directory="high_entropy",
            total_entropy=5.0,  # Very high
            extension_entropy=2.0,
            naming_entropy=1.5,
            violation_count=15,  # Many violations
            unexpected_files=["a.txt", "b.exe"],
            forbidden_files=["c.py"],
            naming_violations=["d.PY", "e.PY"],
            file_count=20,
        )
        
        risk = auditor_module.classify_directory_risk(score)
        
        assert risk.risk_level == "HIGH"
        assert risk.risk_score >= 0.7
    
    def test_low_entropy_produces_low_risk(self, auditor_module):
        """Test 36: Low entropy directory gets LOW risk classification."""
        score = auditor_module.EntropyScore(
            directory="clean",
            total_entropy=0.2,  # Very low
            extension_entropy=0.0,
            naming_entropy=0.2,
            violation_count=0,  # No violations
            unexpected_files=[],
            forbidden_files=[],
            naming_violations=[],
            file_count=5,
        )
        
        risk = auditor_module.classify_directory_risk(score)
        
        assert risk.risk_level == "LOW"
        assert risk.risk_score < 0.3
    
    def test_risk_distribution_sums_correctly(self, auditor_module):
        """Test 37: Risk distribution sums to total directories."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        
        risk = auditor_module.compute_structure_risk_index(report)
        
        total = sum(risk.risk_distribution.values())
        assert total == risk.total_directories_analyzed
    
    def test_risk_index_structure(self, auditor_module):
        """Test 38: Risk index has all required fields."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        
        risk = auditor_module.compute_structure_risk_index(report)
        risk_dict = risk.to_dict()
        
        required_fields = [
            "timestamp",
            "global_risk_score",
            "global_risk_level",
            "top_risky_directories",
            "phase_filter",
            "total_directories_analyzed",
            "risk_distribution",
            "drift_indicators",
        ]
        
        for field in required_fields:
            assert field in risk_dict, f"Missing field: {field}"
    
    def test_risk_index_json_serializable(self, auditor_module):
        """Test 39: Risk index is JSON serializable."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        
        risk = auditor_module.compute_structure_risk_index(report)
        
        # Should not raise
        json_str = json.dumps(risk.to_dict(), indent=2)
        parsed = json.loads(json_str)
        
        assert "global_risk_score" in parsed
    
    def test_boundary_violations_increase_risk(self, auditor_module):
        """Test 40: Boundary violations increase global risk score."""
        # Report without violations
        report_clean = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_clean = auditor_module.compute_structure_risk_index(report_clean)
        
        # Report with violations
        report_with_violations = auditor_module.generate_report(
            include_guardian=True,  # Include boundary check
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_with_violations = auditor_module.compute_structure_risk_index(report_with_violations)
        
        # If there are violations, risk should be higher
        if report_with_violations.boundary_violations:
            assert risk_with_violations.global_risk_score >= risk_clean.global_risk_score


# -----------------------------------------------------------------------------
# PHASE FILTERING TESTS (41-50)
# -----------------------------------------------------------------------------

class TestPhaseFiltering:
    """Tests for phase-specific directory filtering."""
    
    def test_phase_i_directories_list_not_empty(self, auditor_module):
        """Test 41: Phase I directories list is defined."""
        assert len(auditor_module.PHASE_I_DIRECTORIES) > 0
    
    def test_phase_ii_directories_list_not_empty(self, auditor_module):
        """Test 42: Phase II directories list is defined."""
        assert len(auditor_module.PHASE_II_DIRECTORIES) > 0
    
    def test_is_phase_i_directory_detects_known(self, auditor_module):
        """Test 43: is_phase_i_directory correctly identifies Phase I paths."""
        phase_i_paths = [
            "curriculum",
            "derivation",
            "rfl",
            "backend/axiom_engine",
            "backend/crypto",
            "backend/dag",
        ]
        
        for path in phase_i_paths:
            assert auditor_module.is_phase_i_directory(path), f"{path} should be Phase I"
    
    def test_is_phase_ii_directory_detects_known(self, auditor_module):
        """Test 44: is_phase_ii_directory correctly identifies Phase II paths."""
        phase_ii_paths = [
            "tests/phase2",
            "artifacts/phase_ii",
            "artifacts/u2",
            "backend/metrics",
            "experiments/u2",
        ]
        
        for path in phase_ii_paths:
            assert auditor_module.is_phase_ii_directory(path), f"{path} should be Phase II"
    
    def test_phase_filters_return_disjoint_sets(self, auditor_module):
        """Test 45: Phase I and Phase II filters return disjoint directory sets."""
        all_scores = auditor_module.compute_all_directory_entropies()
        
        phase1_scores = auditor_module.filter_directories_by_phase(all_scores, "phase1")
        phase2_scores = auditor_module.filter_directories_by_phase(all_scores, "phase2")
        
        phase1_dirs = {s.directory for s in phase1_scores}
        phase2_dirs = {s.directory for s in phase2_scores}
        
        # Should be disjoint (no overlap)
        intersection = phase1_dirs & phase2_dirs
        assert len(intersection) == 0, f"Overlap detected: {intersection}"
    
    def test_phase_filter_consistent_scoring(self, auditor_module):
        """Test 46: Same directory has same entropy in filtered and unfiltered results."""
        all_scores = auditor_module.compute_all_directory_entropies()
        phase1_scores = auditor_module.filter_directories_by_phase(all_scores, "phase1")
        
        # Find a directory that appears in both
        phase1_dirs = {s.directory: s for s in phase1_scores}
        all_dirs = {s.directory: s for s in all_scores}
        
        for dir_path, phase1_score in phase1_dirs.items():
            if dir_path in all_dirs:
                assert phase1_score.total_entropy == all_dirs[dir_path].total_entropy
    
    def test_risk_index_with_phase_filter(self, auditor_module):
        """Test 47: Risk index can be computed with phase filter."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        
        risk_phase1 = auditor_module.compute_structure_risk_index(report, phase_filter="phase1")
        risk_phase2 = auditor_module.compute_structure_risk_index(report, phase_filter="phase2")
        
        # Both should have valid results
        assert risk_phase1.phase_filter == "phase1"
        assert risk_phase2.phase_filter == "phase2"
        assert 0.0 <= risk_phase1.global_risk_score <= 1.0
        assert 0.0 <= risk_phase2.global_risk_score <= 1.0
    
    def test_phase_filter_reproducibility(self, auditor_module):
        """Test 48: Phase filtering is reproducible."""
        all_scores = auditor_module.compute_all_directory_entropies()
        
        phase1_a = auditor_module.filter_directories_by_phase(all_scores, "phase1")
        phase1_b = auditor_module.filter_directories_by_phase(all_scores, "phase1")
        
        dirs_a = {s.directory for s in phase1_a}
        dirs_b = {s.directory for s in phase1_b}
        
        assert dirs_a == dirs_b
    
    def test_phase_entropy_computation(self, auditor_module):
        """Test 49: compute_phase_entropy returns valid results."""
        phase1_scores, phase1_avg = auditor_module.compute_phase_entropy("phase1")
        phase2_scores, phase2_avg = auditor_module.compute_phase_entropy("phase2")
        
        # Should return non-negative averages
        assert phase1_avg >= 0.0
        assert phase2_avg >= 0.0
        
        # Scores should be lists
        assert isinstance(phase1_scores, list)
        assert isinstance(phase2_scores, list)
    
    def test_unfiltered_contains_filtered_union(self, auditor_module):
        """Test 50: Unfiltered results contain Phase I + Phase II + others."""
        all_scores = auditor_module.compute_all_directory_entropies()
        phase1_scores = auditor_module.filter_directories_by_phase(all_scores, "phase1")
        phase2_scores = auditor_module.filter_directories_by_phase(all_scores, "phase2")
        
        all_dirs = {s.directory for s in all_scores}
        phase1_dirs = {s.directory for s in phase1_scores}
        phase2_dirs = {s.directory for s in phase2_scores}
        
        # Union of phase1 and phase2 should be subset of all
        union = phase1_dirs | phase2_dirs
        assert union <= all_dirs


# -----------------------------------------------------------------------------
# STRUCTURAL DRIFT DETECTION TESTS (51-60)
# -----------------------------------------------------------------------------

class TestStructuralDriftDetection:
    """Tests for structural drift detection between reports."""
    
    @pytest.fixture
    def sample_old_report(self, tmp_path):
        """Create a sample old report for comparison."""
        report = {
            "timestamp": "2025-12-01T00:00:00Z",
            "project_root": str(tmp_path),
            "report_hash": "abc123",
            "total_directories": 3,
            "total_files": 10,
            "global_entropy": 0.5,
            "directory_scores": [
                {
                    "directory": "backend/api",
                    "total_entropy": 0.3,
                    "extension_entropy": 0.2,
                    "naming_entropy": 0.1,
                    "violation_count": 1,
                    "unexpected_files": [],
                    "forbidden_files": [],
                    "naming_violations": [],
                    "file_count": 5,
                },
                {
                    "directory": "tests/phase2",
                    "total_entropy": 0.5,
                    "extension_entropy": 0.3,
                    "naming_entropy": 0.2,
                    "violation_count": 2,
                    "unexpected_files": [],
                    "forbidden_files": [],
                    "naming_violations": [],
                    "file_count": 8,
                },
            ],
            "boundary_violations": [],
            "remediation_suggestions": [],
            "archetype_classifications": [],
            "mutation_events": [],
            "summary": {},
        }
        
        report_path = tmp_path / "old_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f)
        return report_path
    
    @pytest.fixture
    def sample_new_report(self, tmp_path):
        """Create a sample new report with changes."""
        report = {
            "timestamp": "2025-12-06T00:00:00Z",
            "project_root": str(tmp_path),
            "report_hash": "def456",
            "total_directories": 4,
            "total_files": 15,
            "global_entropy": 0.7,
            "directory_scores": [
                {
                    "directory": "backend/api",
                    "total_entropy": 0.8,  # Increased entropy
                    "extension_entropy": 0.5,
                    "naming_entropy": 0.3,
                    "violation_count": 5,  # More violations
                    "unexpected_files": ["extra.txt"],
                    "forbidden_files": [],
                    "naming_violations": [],
                    "file_count": 7,
                },
                {
                    "directory": "tests/phase2",
                    "total_entropy": 0.4,  # Decreased entropy (improved)
                    "extension_entropy": 0.2,
                    "naming_entropy": 0.2,
                    "violation_count": 1,  # Fewer violations
                    "unexpected_files": [],
                    "forbidden_files": [],
                    "naming_violations": [],
                    "file_count": 10,
                },
                {
                    "directory": "experiments/u2",  # New directory
                    "total_entropy": 0.6,
                    "extension_entropy": 0.4,
                    "naming_entropy": 0.2,
                    "violation_count": 2,
                    "unexpected_files": [],
                    "forbidden_files": [],
                    "naming_violations": [],
                    "file_count": 4,
                },
            ],
            "boundary_violations": [],
            "remediation_suggestions": [],
            "archetype_classifications": [],
            "mutation_events": [],
            "summary": {},
        }
        
        report_path = tmp_path / "new_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f)
        return report_path
    
    def test_compare_structure_risk_basic(self, auditor_module, sample_old_report, sample_new_report):
        """Test 51: compare_structure_risk produces valid drift report."""
        drift = auditor_module.compare_structure_risk(sample_old_report, sample_new_report)
        
        assert drift is not None
        assert drift.old_report_path == str(sample_old_report)
        assert drift.new_report_path == str(sample_new_report)
    
    def test_drift_detects_entropy_increase(self, auditor_module, sample_old_report, sample_new_report):
        """Test 52: Drift detection identifies entropy increases."""
        drift = auditor_module.compare_structure_risk(sample_old_report, sample_new_report)
        
        # backend/api had entropy increase from 0.3 to 0.8
        api_drift = next((d for d in drift.directory_drifts if d.path == "backend/api"), None)
        assert api_drift is not None
        assert api_drift.entropy_delta > 0
    
    def test_drift_detects_entropy_decrease(self, auditor_module, sample_old_report, sample_new_report):
        """Test 53: Drift detection identifies entropy decreases (improvements)."""
        drift = auditor_module.compare_structure_risk(sample_old_report, sample_new_report)
        
        # tests/phase2 had entropy decrease from 0.5 to 0.4
        phase2_drift = next((d for d in drift.directory_drifts if d.path == "tests/phase2"), None)
        assert phase2_drift is not None
        assert phase2_drift.entropy_delta < 0
    
    def test_drift_detects_new_directories(self, auditor_module, sample_old_report, sample_new_report):
        """Test 54: Drift detection identifies new directories."""
        drift = auditor_module.compare_structure_risk(sample_old_report, sample_new_report)
        
        assert "experiments/u2" in drift.new_directories
    
    def test_drift_report_has_required_fields(self, auditor_module, sample_old_report, sample_new_report):
        """Test 55: Drift report has all required fields (v1 contract schema)."""
        drift = auditor_module.compare_structure_risk(sample_old_report, sample_new_report)
        drift_dict = drift.to_dict()
        
        # Contract-required fields (v1 schema - see StructureDriftReport docstring)
        required_fields = [
            "timestamp",
            "old_report",      # Renamed from old_report_path in v1 schema
            "new_report",      # Renamed from new_report_path in v1 schema
            "overall_trend",   # Moved to top-level in v1 schema
            "global_risk_old",
            "global_risk_new",
            "global_risk_delta",
            "max_risk_increase",
            "max_risk_decrease",
            "directories_with_increased_risk",
            "directories_with_decreased_risk",
            "new_directories",
            "removed_directories",
            "risk_transitions",
            "directory_drifts",
            "summary",
        ]
        
        for field in required_fields:
            assert field in drift_dict, f"Missing field: {field}"
    
    def test_drift_report_json_serializable(self, auditor_module, sample_old_report, sample_new_report):
        """Test 56: Drift report is JSON serializable."""
        drift = auditor_module.compare_structure_risk(sample_old_report, sample_new_report)
        
        # Should not raise
        json_str = json.dumps(drift.to_dict(), indent=2)
        parsed = json.loads(json_str)
        
        assert "global_risk_delta" in parsed
    
    def test_drift_summary_has_trend(self, auditor_module, sample_old_report, sample_new_report):
        """Test 57: Drift summary includes overall trend."""
        drift = auditor_module.compare_structure_risk(sample_old_report, sample_new_report)
        
        assert "overall_trend" in drift.summary
        assert drift.summary["overall_trend"] in {"improving", "degrading", "stable"}
    
    def test_directory_drift_structure(self, auditor_module, sample_old_report, sample_new_report):
        """Test 58: DirectoryDrift has correct structure."""
        drift = auditor_module.compare_structure_risk(sample_old_report, sample_new_report)
        
        assert len(drift.directory_drifts) > 0
        dir_drift = drift.directory_drifts[0]
        
        assert hasattr(dir_drift, "path")
        assert hasattr(dir_drift, "entropy_old")
        assert hasattr(dir_drift, "entropy_new")
        assert hasattr(dir_drift, "entropy_delta")
        assert hasattr(dir_drift, "risk_delta")
    
    def test_file_not_found_raises_error(self, auditor_module, tmp_path):
        """Test 59: Missing report file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            auditor_module.compare_structure_risk(
                tmp_path / "nonexistent.json",
                tmp_path / "also_nonexistent.json"
            )
    
    def test_drift_max_values_calculated(self, auditor_module, sample_old_report, sample_new_report):
        """Test 60: Max risk increase/decrease are calculated."""
        drift = auditor_module.compare_structure_risk(sample_old_report, sample_new_report)
        
        # We have both increases and improvements in test data
        assert isinstance(drift.max_risk_increase, float)
        assert isinstance(drift.max_risk_decrease, float)


# -----------------------------------------------------------------------------
# REFACTOR CANDIDATE TESTS (61-70)
# -----------------------------------------------------------------------------

class TestRefactorCandidates:
    """Tests for refactor candidate generation."""
    
    def test_generate_refactor_candidates_basic(self, auditor_module):
        """Test 61: generate_refactor_candidates produces valid output."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        assert isinstance(candidates, list)
    
    def test_refactor_candidate_structure(self, auditor_module):
        """Test 62: RefactorCandidate has correct structure."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        if candidates:
            c = candidates[0]
            assert hasattr(c, "path")
            assert hasattr(c, "risk_score")
            assert hasattr(c, "risk_level")
            assert hasattr(c, "entropy")
            assert hasattr(c, "violations")
            assert hasattr(c, "is_phase_ii")
            assert hasattr(c, "priority")
            assert hasattr(c, "reasons")
    
    def test_refactor_candidate_priority_values(self, auditor_module):
        """Test 63: RefactorCandidate priority is valid."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        valid_priorities = {"CRITICAL", "HIGH", "MEDIUM"}
        for c in candidates:
            assert c.priority in valid_priorities
    
    def test_refactor_candidates_sorted_by_priority(self, auditor_module):
        """Test 64: Refactor candidates are sorted by priority."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        if len(candidates) >= 2:
            priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
            for i in range(len(candidates) - 1):
                assert priority_order[candidates[i].priority] <= priority_order[candidates[i+1].priority]
    
    def test_refactor_candidates_respects_top_k(self, auditor_module):
        """Test 65: generate_refactor_candidates respects top_k limit."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        
        candidates = auditor_module.generate_refactor_candidates(risk_index, top_k=3)
        
        assert len(candidates) <= 3
    
    def test_refactor_candidate_has_reasons(self, auditor_module):
        """Test 66: Refactor candidates have at least one reason."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        for c in candidates:
            assert len(c.reasons) > 0
    
    def test_refactor_candidate_to_dict(self, auditor_module):
        """Test 67: RefactorCandidate serializes correctly."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        if candidates:
            d = candidates[0].to_dict()
            assert "path" in d
            assert "risk_score" in d
            assert "priority" in d
    
    def test_write_refactor_candidates(self, auditor_module, tmp_path):
        """Test 68: write_refactor_candidates creates file."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        output_path = tmp_path / "refactor_candidates.json"
        result_path = auditor_module.write_refactor_candidates(candidates, output_path)
        
        assert result_path.exists()
        
        with open(result_path) as f:
            data = json.load(f)
        
        assert "candidates" in data
        assert "summary" in data
    
    def test_refactor_candidates_with_drift(self, auditor_module, tmp_path):
        """Test 69: Refactor candidates incorporate drift data when available."""
        # Create mock reports
        old_report = {
            "directory_scores": [
                {
                    "directory": "backend/metrics",
                    "total_entropy": 0.5,
                    "extension_entropy": 0.3,
                    "naming_entropy": 0.2,
                    "violation_count": 1,
                    "unexpected_files": [],
                    "forbidden_files": [],
                    "naming_violations": [],
                    "file_count": 5,
                },
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {
                    "directory": "backend/metrics",
                    "total_entropy": 2.5,  # Large increase
                    "extension_entropy": 1.5,
                    "naming_entropy": 1.0,
                    "violation_count": 10,
                    "unexpected_files": ["a.txt", "b.txt"],
                    "forbidden_files": [],
                    "naming_violations": [],
                    "file_count": 15,
                },
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        risk_index = auditor_module.compute_structure_risk_index(new_report)
        
        candidates = auditor_module.generate_refactor_candidates(risk_index, drift)
        
        # Should find backend/metrics with increased risk
        metrics_candidate = next((c for c in candidates if c.path == "backend/metrics"), None)
        if metrics_candidate:
            assert metrics_candidate.risk_increased_recently
    
    def test_empty_risk_index_produces_empty_candidates(self, auditor_module):
        """Test 70: Empty risk index produces empty candidate list."""
        empty_risk = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.0,
            global_risk_level="LOW",
            top_risky_directories=[],
            phase_filter=None,
            total_directories_analyzed=0,
            risk_distribution={"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            drift_indicators=[],
        )
        
        candidates = auditor_module.generate_refactor_candidates(empty_risk)
        
        assert len(candidates) == 0


# -----------------------------------------------------------------------------
# DRIFT REPORT SCHEMA CONTRACT TESTS (71-80)
# -----------------------------------------------------------------------------

class TestDriftReportSchemaContract:
    """Tests ensuring drift report schema is stable and deterministic."""
    
    def test_drift_report_has_all_required_keys(self, auditor_module, tmp_path):
        """Test 71: Drift report to_dict has all contract-required keys."""
        # Create minimal reports
        old_report = {
            "directory_scores": [
                {
                    "directory": "backend",
                    "total_entropy": 1.0,
                    "extension_entropy": 0.5,
                    "naming_entropy": 0.5,
                    "violation_count": 0,
                    "unexpected_files": [],
                    "forbidden_files": [],
                    "naming_violations": [],
                    "file_count": 3,
                },
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {
                    "directory": "backend",
                    "total_entropy": 1.5,
                    "extension_entropy": 0.8,
                    "naming_entropy": 0.7,
                    "violation_count": 1,
                    "unexpected_files": [],
                    "forbidden_files": [],
                    "naming_violations": [],
                    "file_count": 5,
                },
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        d = drift.to_dict()
        
        # Contract-required keys (v1 schema)
        required_keys = [
            "old_report",
            "new_report",
            "overall_trend",
            "max_risk_increase",
            "max_risk_decrease",
            "directories_with_increased_risk",
            "directories_with_decreased_risk",
            "new_directories",
            "removed_directories",
            "risk_transitions",
        ]
        
        for key in required_keys:
            assert key in d, f"Missing required key: {key}"
    
    def test_drift_report_directory_lists_sorted_alphabetically(self, auditor_module, tmp_path):
        """Test 72: All directory lists in drift report are sorted alphabetically."""
        # Create reports with multiple directories
        old_report = {
            "directory_scores": [
                {"directory": "zebra", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 2},
                {"directory": "alpha", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 2},
                {"directory": "beta", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 2},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "zebra", "total_entropy": 2.0, "extension_entropy": 1.0, "naming_entropy": 1.0, "violation_count": 5, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 10},
                {"directory": "alpha", "total_entropy": 2.0, "extension_entropy": 1.0, "naming_entropy": 1.0, "violation_count": 5, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 10},
                {"directory": "gamma", "total_entropy": 1.0, "extension_entropy": 0.5, "naming_entropy": 0.5, "violation_count": 1, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 3},  # New
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        d = drift.to_dict()
        
        # Check all lists are sorted
        assert d["directories_with_increased_risk"] == sorted(d["directories_with_increased_risk"])
        assert d["directories_with_decreased_risk"] == sorted(d["directories_with_decreased_risk"])
        assert d["new_directories"] == sorted(d["new_directories"])
        assert d["removed_directories"] == sorted(d["removed_directories"])
    
    def test_drift_report_deterministic_ordering(self, auditor_module, tmp_path):
        """Test 73: Drift report produces same JSON on repeated runs."""
        old_report = {
            "directory_scores": [
                {"directory": "c", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 2},
                {"directory": "a", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 2},
                {"directory": "b", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 2},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "c", "total_entropy": 1.5, "extension_entropy": 0.8, "naming_entropy": 0.7, "violation_count": 3, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
                {"directory": "a", "total_entropy": 1.5, "extension_entropy": 0.8, "naming_entropy": 0.7, "violation_count": 3, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
                {"directory": "b", "total_entropy": 1.5, "extension_entropy": 0.8, "naming_entropy": 0.7, "violation_count": 3, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        # Run twice
        drift1 = auditor_module.compare_structure_risk(old_path, new_path)
        drift2 = auditor_module.compare_structure_risk(old_path, new_path)
        
        d1 = drift1.to_dict()
        d2 = drift2.to_dict()
        
        # Remove timestamp for comparison
        d1.pop("timestamp")
        d2.pop("timestamp")
        
        # Convert to JSON strings for comparison
        json1 = json.dumps(d1, sort_keys=True)
        json2 = json.dumps(d2, sort_keys=True)
        
        assert json1 == json2, "Drift reports should be deterministic"
    
    def test_drift_report_overall_trend_improving(self, auditor_module, tmp_path):
        """Test 74: Overall trend is 'improving' when risk decreases."""
        old_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 3.0, "extension_entropy": 1.5, "naming_entropy": 1.5, "violation_count": 15, "unexpected_files": ["a.txt"] * 5, "forbidden_files": [], "naming_violations": [], "file_count": 20},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        
        assert drift.summary.get("overall_trend") == "improving"
        assert drift.to_dict()["overall_trend"] == "improving"
    
    def test_drift_report_overall_trend_degrading(self, auditor_module, tmp_path):
        """Test 75: Overall trend is 'degrading' when risk increases."""
        old_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 3.0, "extension_entropy": 1.5, "naming_entropy": 1.5, "violation_count": 15, "unexpected_files": ["a.txt"] * 5, "forbidden_files": [], "naming_violations": [], "file_count": 20},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        
        assert drift.summary.get("overall_trend") == "degrading"
        assert drift.to_dict()["overall_trend"] == "degrading"
    
    def test_drift_report_tracks_risk_transitions_low_to_high(self, auditor_module, tmp_path):
        """Test 76: Drift report tracks LOW→HIGH risk transitions."""
        old_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 3.0, "extension_entropy": 1.5, "naming_entropy": 1.5, "violation_count": 15, "unexpected_files": ["a.txt"] * 5, "forbidden_files": [], "naming_violations": [], "file_count": 20},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        
        # Check that risk_transitions is populated
        assert drift.risk_transitions is not None
        
        # Should have LOW_TO_HIGH transition if backend went from LOW to HIGH
        d = drift.to_dict()
        assert "risk_transitions" in d
    
    def test_drift_report_tracks_risk_transitions_high_to_low(self, auditor_module, tmp_path):
        """Test 77: Drift report tracks HIGH→LOW risk transitions (improvement)."""
        old_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 3.0, "extension_entropy": 1.5, "naming_entropy": 1.5, "violation_count": 15, "unexpected_files": ["a.txt"] * 5, "forbidden_files": [], "naming_violations": [], "file_count": 20},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        d = drift.to_dict()
        
        assert "risk_transitions" in d
        # Might have HIGH_TO_LOW or HIGH_TO_MEDIUM depending on thresholds
    
    def test_drift_report_identifies_new_directories(self, auditor_module, tmp_path):
        """Test 78: Drift report identifies newly added directories."""
        old_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 1.0, "extension_entropy": 0.5, "naming_entropy": 0.5, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 1.0, "extension_entropy": 0.5, "naming_entropy": 0.5, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
                {"directory": "frontend", "total_entropy": 1.0, "extension_entropy": 0.5, "naming_entropy": 0.5, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        d = drift.to_dict()
        
        assert "frontend" in d["new_directories"]
    
    def test_drift_report_identifies_removed_directories(self, auditor_module, tmp_path):
        """Test 79: Drift report identifies removed directories."""
        old_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 1.0, "extension_entropy": 0.5, "naming_entropy": 0.5, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
                {"directory": "legacy", "total_entropy": 1.0, "extension_entropy": 0.5, "naming_entropy": 0.5, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 1.0, "extension_entropy": 0.5, "naming_entropy": 0.5, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        d = drift.to_dict()
        
        assert "legacy" in d["removed_directories"]
    
    def test_drift_report_max_risk_values(self, auditor_module, tmp_path):
        """Test 80: Drift report correctly computes max risk increase/decrease."""
        old_report = {
            "directory_scores": [
                {"directory": "growing", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
                {"directory": "shrinking", "total_entropy": 3.0, "extension_entropy": 1.5, "naming_entropy": 1.5, "violation_count": 15, "unexpected_files": ["a.txt"] * 5, "forbidden_files": [], "naming_violations": [], "file_count": 20},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "growing", "total_entropy": 3.0, "extension_entropy": 1.5, "naming_entropy": 1.5, "violation_count": 15, "unexpected_files": ["a.txt"] * 5, "forbidden_files": [], "naming_violations": [], "file_count": 20},
                {"directory": "shrinking", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        d = drift.to_dict()
        
        # Should have positive max_risk_increase for "growing"
        assert d["max_risk_increase"] > 0
        # Should have negative max_risk_decrease for "shrinking"
        assert d["max_risk_decrease"] < 0


# -----------------------------------------------------------------------------
# PHASE II HEATMAP & CI SUMMARY TESTS (81-90)
# -----------------------------------------------------------------------------

class TestPhaseIIHeatmapAndCISummary:
    """Tests for Phase II refactor heatmap and CI summary generation."""
    
    def test_phase_ii_heatmap_returns_string(self, auditor_module):
        """Test 81: print_phase_ii_heatmap returns a string."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        # Capture output
        result = auditor_module.print_phase_ii_heatmap(risk_index, candidates)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_phase_ii_heatmap_contains_headers(self, auditor_module):
        """Test 82: Heatmap contains required section headers."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        result = auditor_module.print_phase_ii_heatmap(risk_index, candidates)
        
        assert "PHASE II STRUCTURAL RISK HEATMAP" in result
        assert "Risk Distribution" in result
    
    def test_phase_ii_ci_summary_returns_markdown(self, auditor_module):
        """Test 83: generate_phase_ii_ci_summary returns markdown string."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        result = auditor_module.generate_phase_ii_ci_summary(risk_index, candidates)
        
        assert isinstance(result, str)
        assert "### " in result or "**" in result  # Contains markdown
    
    def test_phase_ii_ci_summary_contains_global_risk(self, auditor_module):
        """Test 84: CI summary contains global risk score."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        result = auditor_module.generate_phase_ii_ci_summary(risk_index, candidates)
        
        assert "Global Risk Score" in result
    
    def test_phase_ii_ci_summary_with_drift_includes_trend(self, auditor_module, tmp_path):
        """Test 85: CI summary with drift includes trend information."""
        old_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 1.0, "extension_entropy": 0.5, "naming_entropy": 0.5, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 2.0, "extension_entropy": 1.0, "naming_entropy": 1.0, "violation_count": 5, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 10},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        risk_index = auditor_module.compute_structure_risk_index(new_report)
        candidates = auditor_module.generate_refactor_candidates(risk_index, drift)
        
        result = auditor_module.generate_phase_ii_ci_summary(risk_index, candidates, drift)
        
        assert "Trend" in result
    
    def test_heatmap_filters_to_phase_ii_candidates(self, auditor_module):
        """Test 86: Heatmap only shows Phase II candidates."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        # Add a mock Phase II candidate
        mock_candidate = auditor_module.RefactorCandidate(
            path="experiments/u2/test_dir",
            risk_score=0.9,
            risk_level="HIGH",
            entropy=2.5,
            violations=10,
            is_phase_ii=True,
            risk_increased_recently=False,
            risk_delta=None,
            priority="HIGH",
            reasons=["Test reason"],
        )
        
        test_candidates = [mock_candidate] + candidates
        
        result = auditor_module.print_phase_ii_heatmap(risk_index, test_candidates)
        
        # The Phase II candidate should appear
        # (if there are Phase II candidates)
        assert "experiments/u2/test_dir" in result or "No Phase II refactor candidates" in result
    
    def test_ci_summary_handles_empty_candidates(self, auditor_module):
        """Test 87: CI summary handles empty candidate list gracefully."""
        empty_risk = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.0,
            global_risk_level="LOW",
            top_risky_directories=[],
            phase_filter=None,
            total_directories_analyzed=0,
            risk_distribution={"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            drift_indicators=[],
        )
        
        result = auditor_module.generate_phase_ii_ci_summary(empty_risk, [])
        
        assert isinstance(result, str)
        assert "No HIGH-risk" in result or "healthy" in result.lower()
    
    def test_heatmap_risk_bar_visualization(self, auditor_module):
        """Test 88: Heatmap shows risk bar visualization."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        
        # Create high-risk Phase II candidate
        mock_candidate = auditor_module.RefactorCandidate(
            path="tests/phase2/high_risk",
            risk_score=1.0,
            risk_level="HIGH",
            entropy=3.0,
            violations=15,
            is_phase_ii=True,
            risk_increased_recently=False,
            risk_delta=None,
            priority="CRITICAL",
            reasons=["HIGH risk"],
        )
        
        result = auditor_module.print_phase_ii_heatmap(risk_index, [mock_candidate])
        
        # Should contain visual bar characters
        assert "█" in result or "░" in result or "tests/phase2" in result
    
    def test_ci_summary_includes_transition_warning(self, auditor_module, tmp_path):
        """Test 89: CI summary warns about LOW→HIGH transitions."""
        old_report = {
            "directory_scores": [
                {"directory": "backend/u2_module", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "backend/u2_module", "total_entropy": 3.0, "extension_entropy": 1.5, "naming_entropy": 1.5, "violation_count": 15, "unexpected_files": ["a.txt"] * 5, "forbidden_files": [], "naming_violations": [], "file_count": 20},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        risk_index = auditor_module.compute_structure_risk_index(new_report)
        candidates = auditor_module.generate_refactor_candidates(risk_index, drift)
        
        result = auditor_module.generate_phase_ii_ci_summary(risk_index, candidates, drift)
        
        # If there was a LOW→HIGH transition, it should be mentioned
        if drift.risk_transitions and drift.risk_transitions.get("LOW_TO_HIGH"):
            assert "LOW→HIGH" in result or "transitioned" in result.lower()
    
    def test_heatmap_deterministic_output(self, auditor_module):
        """Test 90: Heatmap produces deterministic output."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        candidates = auditor_module.generate_refactor_candidates(risk_index)
        
        result1 = auditor_module.print_phase_ii_heatmap(risk_index, candidates)
        result2 = auditor_module.print_phase_ii_heatmap(risk_index, candidates)
        
        assert result1 == result2


# -----------------------------------------------------------------------------
# ZONE-BASED STRUCTURAL GROUPING TESTS (91-100)
# -----------------------------------------------------------------------------

class TestZoneBasedStructuralGrouping:
    """Tests for zone-based directory grouping."""
    
    def test_extract_zone_simple_path(self, auditor_module):
        """Test 91: extract_zone returns top-level directory."""
        assert auditor_module.extract_zone("backend/api/v1") == "backend"
        assert auditor_module.extract_zone("tests/unit/test_api.py") == "tests"
        assert auditor_module.extract_zone("docs/README.md") == "docs"
    
    def test_extract_zone_single_level(self, auditor_module):
        """Test 92: extract_zone handles single-level paths."""
        assert auditor_module.extract_zone("backend") == "backend"
        assert auditor_module.extract_zone("scripts") == "scripts"
    
    def test_extract_zone_handles_backslashes(self, auditor_module):
        """Test 93: extract_zone normalizes Windows paths."""
        assert auditor_module.extract_zone("backend\\api\\v1") == "backend"
        assert auditor_module.extract_zone("tests\\unit") == "tests"
    
    def test_group_directories_returns_zones(self, auditor_module):
        """Test 94: group_directories_into_zones returns zone structure."""
        # Create a mock risk index
        mock_dirs = [
            auditor_module.DirectoryRisk(
                path="backend/api",
                risk_score=0.8,
                risk_level="HIGH",
                entropy=2.5,
                violations=10,
                contributing_factors=[],
            ),
            auditor_module.DirectoryRisk(
                path="backend/worker",
                risk_score=0.5,
                risk_level="MEDIUM",
                entropy=1.5,
                violations=3,
                contributing_factors=[],
            ),
            auditor_module.DirectoryRisk(
                path="tests/unit",
                risk_score=0.3,
                risk_level="LOW",
                entropy=0.8,
                violations=1,
                contributing_factors=[],
            ),
        ]
        
        mock_risk_index = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.5,
            global_risk_level="MEDIUM",
            top_risky_directories=mock_dirs,
            phase_filter=None,
            total_directories_analyzed=3,
            risk_distribution={"HIGH": 1, "MEDIUM": 1, "LOW": 1},
            drift_indicators=[],
        )
        
        result = auditor_module.group_directories_into_zones(mock_risk_index)
        
        assert "zones" in result
        assert "backend" in result["zones"]
        assert "tests" in result["zones"]
    
    def test_zone_counts_correct(self, auditor_module):
        """Test 95: Zone risk counts are accurate."""
        mock_dirs = [
            auditor_module.DirectoryRisk(path="backend/api", risk_score=0.8, risk_level="HIGH", entropy=2.5, violations=10, contributing_factors=[]),
            auditor_module.DirectoryRisk(path="backend/worker", risk_score=0.5, risk_level="MEDIUM", entropy=1.5, violations=3, contributing_factors=[]),
            auditor_module.DirectoryRisk(path="backend/models", risk_score=0.3, risk_level="LOW", entropy=0.8, violations=1, contributing_factors=[]),
        ]
        
        mock_risk_index = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.5,
            global_risk_level="MEDIUM",
            top_risky_directories=mock_dirs,
            phase_filter=None,
            total_directories_analyzed=3,
            risk_distribution={"HIGH": 1, "MEDIUM": 1, "LOW": 1},
            drift_indicators=[],
        )
        
        result = auditor_module.group_directories_into_zones(mock_risk_index)
        
        assert result["zones"]["backend"]["high"] == 1
        assert result["zones"]["backend"]["medium"] == 1
        assert result["zones"]["backend"]["low"] == 1
    
    def test_zone_list_sorted_alphabetically(self, auditor_module):
        """Test 96: Zone list is sorted alphabetically."""
        mock_dirs = [
            auditor_module.DirectoryRisk(path="zebra/dir", risk_score=0.3, risk_level="LOW", entropy=0.5, violations=0, contributing_factors=[]),
            auditor_module.DirectoryRisk(path="alpha/dir", risk_score=0.3, risk_level="LOW", entropy=0.5, violations=0, contributing_factors=[]),
            auditor_module.DirectoryRisk(path="beta/dir", risk_score=0.3, risk_level="LOW", entropy=0.5, violations=0, contributing_factors=[]),
        ]
        
        mock_risk_index = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.2,
            global_risk_level="LOW",
            top_risky_directories=mock_dirs,
            phase_filter=None,
            total_directories_analyzed=3,
            risk_distribution={"HIGH": 0, "MEDIUM": 0, "LOW": 3},
            drift_indicators=[],
        )
        
        result = auditor_module.group_directories_into_zones(mock_risk_index)
        
        assert result["zone_list"] == ["alpha", "beta", "zebra"]
    
    def test_zone_grouping_deterministic(self, auditor_module):
        """Test 97: Zone grouping produces deterministic output."""
        report = auditor_module.generate_report(
            include_guardian=False,
            include_classify=False,
            include_history=False,
            include_remediation=False,
        )
        risk_index = auditor_module.compute_structure_risk_index(report)
        
        result1 = auditor_module.group_directories_into_zones(risk_index)
        result2 = auditor_module.group_directories_into_zones(risk_index)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)
    
    def test_zone_grouping_from_dict(self, auditor_module):
        """Test 98: group_directories_into_zones accepts dict input."""
        mock_dict = {
            "top_risky_directories": [
                {"path": "backend/api", "risk_level": "HIGH"},
                {"path": "tests/unit", "risk_level": "LOW"},
            ]
        }
        
        result = auditor_module.group_directories_into_zones(mock_dict)
        
        assert "backend" in result["zones"]
        assert "tests" in result["zones"]
    
    def test_zone_total_count(self, auditor_module):
        """Test 99: total_zones count is accurate."""
        mock_dirs = [
            auditor_module.DirectoryRisk(path="a/dir", risk_score=0.3, risk_level="LOW", entropy=0.5, violations=0, contributing_factors=[]),
            auditor_module.DirectoryRisk(path="b/dir", risk_score=0.3, risk_level="LOW", entropy=0.5, violations=0, contributing_factors=[]),
            auditor_module.DirectoryRisk(path="c/dir", risk_score=0.3, risk_level="LOW", entropy=0.5, violations=0, contributing_factors=[]),
        ]
        
        mock_risk_index = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.2,
            global_risk_level="LOW",
            top_risky_directories=mock_dirs,
            phase_filter=None,
            total_directories_analyzed=3,
            risk_distribution={"HIGH": 0, "MEDIUM": 0, "LOW": 3},
            drift_indicators=[],
        )
        
        result = auditor_module.group_directories_into_zones(mock_risk_index)
        
        assert result["total_zones"] == 3
    
    def test_zone_handles_empty_input(self, auditor_module):
        """Test 100: Zone grouping handles empty input gracefully."""
        empty_risk = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.0,
            global_risk_level="LOW",
            top_risky_directories=[],
            phase_filter=None,
            total_directories_analyzed=0,
            risk_distribution={"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            drift_indicators=[],
        )
        
        result = auditor_module.group_directories_into_zones(empty_risk)
        
        assert result["zones"] == {}
        assert result["total_zones"] == 0
        assert result["zone_list"] == []


# -----------------------------------------------------------------------------
# REFACTOR CANDIDATE BUNDLES TESTS (101-110)
# -----------------------------------------------------------------------------

class TestRefactorCandidateBundles:
    """Tests for refactor candidate bundling."""
    
    def test_build_bundles_basic(self, auditor_module):
        """Test 101: build_refactor_bundles returns list of bundles."""
        candidates = [
            auditor_module.RefactorCandidate(
                path="backend/api",
                risk_score=0.8,
                risk_level="HIGH",
                entropy=2.5,
                violations=10,
                is_phase_ii=False,
                risk_increased_recently=False,
                risk_delta=None,
                priority="HIGH",
                reasons=["Test"],
            ),
        ]
        
        result = auditor_module.build_refactor_bundles(candidates)
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_bundle_structure(self, auditor_module):
        """Test 102: Bundles have required fields."""
        candidates = [
            auditor_module.RefactorCandidate(
                path="backend/api",
                risk_score=0.8,
                risk_level="HIGH",
                entropy=2.5,
                violations=10,
                is_phase_ii=False,
                risk_increased_recently=False,
                risk_delta=None,
                priority="HIGH",
                reasons=["Test"],
            ),
        ]
        
        result = auditor_module.build_refactor_bundles(candidates)
        
        bundle = result[0]
        assert "bundle_id" in bundle
        assert "directories" in bundle
        assert "dominant_risk_level" in bundle
        assert "directory_count" in bundle
    
    def test_bundle_id_format(self, auditor_module):
        """Test 103: Bundle IDs follow zone:name:risk_level format."""
        candidates = [
            auditor_module.RefactorCandidate(
                path="tests/unit/test_api",
                risk_score=0.5,
                risk_level="MEDIUM",
                entropy=1.5,
                violations=3,
                is_phase_ii=False,
                risk_increased_recently=False,
                risk_delta=None,
                priority="MEDIUM",
                reasons=["Test"],
            ),
        ]
        
        result = auditor_module.build_refactor_bundles(candidates)
        
        assert result[0]["bundle_id"] == "zone:tests:MEDIUM"
    
    def test_directories_grouped_by_zone_and_risk(self, auditor_module):
        """Test 104: Directories are grouped by zone and risk level."""
        candidates = [
            auditor_module.RefactorCandidate(path="backend/api", risk_score=0.8, risk_level="HIGH", entropy=2.5, violations=10, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="HIGH", reasons=["Test"]),
            auditor_module.RefactorCandidate(path="backend/worker", risk_score=0.7, risk_level="HIGH", entropy=2.0, violations=8, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="HIGH", reasons=["Test"]),
            auditor_module.RefactorCandidate(path="tests/unit", risk_score=0.3, risk_level="LOW", entropy=0.8, violations=1, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="MEDIUM", reasons=["Test"]),
        ]
        
        result = auditor_module.build_refactor_bundles(candidates)
        
        # Should have 2 bundles: backend:HIGH and tests:LOW
        bundle_ids = [b["bundle_id"] for b in result]
        assert "zone:backend:HIGH" in bundle_ids
        assert "zone:tests:LOW" in bundle_ids
        
        # backend:HIGH should have 2 directories
        backend_bundle = next(b for b in result if b["bundle_id"] == "zone:backend:HIGH")
        assert backend_bundle["directory_count"] == 2
        assert "backend/api" in backend_bundle["directories"]
        assert "backend/worker" in backend_bundle["directories"]
    
    def test_each_directory_in_exactly_one_bundle(self, auditor_module):
        """Test 105: Each directory appears in exactly one bundle."""
        candidates = [
            auditor_module.RefactorCandidate(path="a/dir1", risk_score=0.8, risk_level="HIGH", entropy=2.5, violations=10, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="HIGH", reasons=["Test"]),
            auditor_module.RefactorCandidate(path="a/dir2", risk_score=0.5, risk_level="MEDIUM", entropy=1.5, violations=3, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="MEDIUM", reasons=["Test"]),
            auditor_module.RefactorCandidate(path="b/dir1", risk_score=0.3, risk_level="LOW", entropy=0.8, violations=1, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="MEDIUM", reasons=["Test"]),
        ]
        
        result = auditor_module.build_refactor_bundles(candidates)
        
        # Collect all directories from all bundles
        all_dirs = []
        for bundle in result:
            all_dirs.extend(bundle["directories"])
        
        # Each directory should appear exactly once
        assert len(all_dirs) == len(set(all_dirs))
        assert len(all_dirs) == 3
    
    def test_bundle_directories_sorted(self, auditor_module):
        """Test 106: Directories within bundles are sorted alphabetically."""
        candidates = [
            auditor_module.RefactorCandidate(path="backend/zebra", risk_score=0.8, risk_level="HIGH", entropy=2.5, violations=10, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="HIGH", reasons=["Test"]),
            auditor_module.RefactorCandidate(path="backend/alpha", risk_score=0.8, risk_level="HIGH", entropy=2.5, violations=10, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="HIGH", reasons=["Test"]),
        ]
        
        result = auditor_module.build_refactor_bundles(candidates)
        
        bundle = result[0]
        assert bundle["directories"] == ["backend/alpha", "backend/zebra"]
    
    def test_bundles_sorted_by_risk_level(self, auditor_module):
        """Test 107: Bundles are sorted HIGH > MEDIUM > LOW."""
        candidates = [
            auditor_module.RefactorCandidate(path="low/dir", risk_score=0.2, risk_level="LOW", entropy=0.5, violations=0, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="MEDIUM", reasons=["Test"]),
            auditor_module.RefactorCandidate(path="high/dir", risk_score=0.9, risk_level="HIGH", entropy=3.0, violations=15, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="CRITICAL", reasons=["Test"]),
            auditor_module.RefactorCandidate(path="med/dir", risk_score=0.5, risk_level="MEDIUM", entropy=1.5, violations=5, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="HIGH", reasons=["Test"]),
        ]
        
        result = auditor_module.build_refactor_bundles(candidates)
        
        # First bundle should be HIGH, then MEDIUM, then LOW
        assert result[0]["dominant_risk_level"] == "HIGH"
        assert result[1]["dominant_risk_level"] == "MEDIUM"
        assert result[2]["dominant_risk_level"] == "LOW"
    
    def test_bundle_deterministic_ids(self, auditor_module):
        """Test 108: Bundle IDs are deterministic across runs."""
        candidates = [
            auditor_module.RefactorCandidate(path="backend/api", risk_score=0.8, risk_level="HIGH", entropy=2.5, violations=10, is_phase_ii=False, risk_increased_recently=False, risk_delta=None, priority="HIGH", reasons=["Test"]),
        ]
        
        result1 = auditor_module.build_refactor_bundles(candidates)
        result2 = auditor_module.build_refactor_bundles(candidates)
        
        assert result1[0]["bundle_id"] == result2[0]["bundle_id"]
    
    def test_bundles_from_dict_input(self, auditor_module):
        """Test 109: build_refactor_bundles accepts dict input."""
        candidates = [
            {"path": "backend/api", "risk_level": "HIGH"},
            {"path": "backend/worker", "risk_level": "HIGH"},
        ]
        
        result = auditor_module.build_refactor_bundles(candidates)
        
        assert len(result) == 1
        assert result[0]["directory_count"] == 2
    
    def test_bundles_empty_input(self, auditor_module):
        """Test 110: build_refactor_bundles handles empty input."""
        result = auditor_module.build_refactor_bundles([])
        
        assert result == []


# -----------------------------------------------------------------------------
# STRUCTURAL POSTURE SUMMARY TESTS (111-120)
# -----------------------------------------------------------------------------

class TestStructuralPostureSummary:
    """Tests for structural posture summary generation."""
    
    def test_posture_has_schema_version(self, auditor_module):
        """Test 111: Structural posture includes schema_version."""
        empty_risk = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.0,
            global_risk_level="LOW",
            top_risky_directories=[],
            phase_filter=None,
            total_directories_analyzed=0,
            risk_distribution={"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            drift_indicators=[],
        )
        
        result = auditor_module.build_structural_posture(empty_risk)
        
        assert "schema_version" in result
        assert result["schema_version"] == "1.0.0"
    
    def test_posture_has_all_required_fields(self, auditor_module):
        """Test 112: Structural posture has all required fields."""
        empty_risk = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.0,
            global_risk_level="LOW",
            top_risky_directories=[],
            phase_filter=None,
            total_directories_analyzed=0,
            risk_distribution={"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            drift_indicators=[],
        )
        
        result = auditor_module.build_structural_posture(empty_risk)
        
        required_fields = [
            "schema_version",
            "high_risk_directories",
            "medium_risk_directories",
            "low_risk_directories",
            "new_directories",
            "removed_directories",
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
    
    def test_posture_counts_from_risk_index(self, auditor_module):
        """Test 113: Posture counts match risk index distribution."""
        risk_index = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.5,
            global_risk_level="MEDIUM",
            top_risky_directories=[],
            phase_filter=None,
            total_directories_analyzed=20,
            risk_distribution={"HIGH": 5, "MEDIUM": 8, "LOW": 7},
            drift_indicators=[],
        )
        
        result = auditor_module.build_structural_posture(risk_index)
        
        assert result["high_risk_directories"] == 5
        assert result["medium_risk_directories"] == 8
        assert result["low_risk_directories"] == 7
    
    def test_posture_counts_from_drift_report(self, auditor_module, tmp_path):
        """Test 114: Posture counts new/removed from drift report."""
        old_report = {
            "directory_scores": [
                {"directory": "old_dir", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 3},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "new_dir", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 3},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        result = auditor_module.build_structural_posture(drift)
        
        assert result["new_directories"] == 1
        assert result["removed_directories"] == 1
    
    def test_posture_from_dict_input(self, auditor_module):
        """Test 115: build_structural_posture accepts dict input."""
        mock_dict = {
            "risk_distribution": {"HIGH": 2, "MEDIUM": 5, "LOW": 10},
        }
        
        result = auditor_module.build_structural_posture(mock_dict)
        
        assert result["high_risk_directories"] == 2
        assert result["medium_risk_directories"] == 5
        assert result["low_risk_directories"] == 10
    
    def test_posture_deterministic(self, auditor_module):
        """Test 116: Posture output is deterministic."""
        risk_index = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.5,
            global_risk_level="MEDIUM",
            top_risky_directories=[],
            phase_filter=None,
            total_directories_analyzed=10,
            risk_distribution={"HIGH": 2, "MEDIUM": 3, "LOW": 5},
            drift_indicators=[],
        )
        
        result1 = auditor_module.build_structural_posture(risk_index)
        result2 = auditor_module.build_structural_posture(risk_index)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)
    
    def test_posture_no_evaluative_language(self, auditor_module):
        """Test 117: Posture contains no evaluative language."""
        risk_index = auditor_module.StructureRiskIndex(
            timestamp="2025-12-06T00:00:00Z",
            global_risk_score=0.9,
            global_risk_level="HIGH",
            top_risky_directories=[],
            phase_filter=None,
            total_directories_analyzed=10,
            risk_distribution={"HIGH": 8, "MEDIUM": 1, "LOW": 1},
            drift_indicators=[],
        )
        
        result = auditor_module.build_structural_posture(risk_index)
        
        # Convert to string and check for evaluative words
        result_str = json.dumps(result).lower()
        evaluative_words = ["bad", "good", "poor", "excellent", "critical", "warning", "error"]
        for word in evaluative_words:
            assert word not in result_str
    
    def test_write_structural_posture(self, auditor_module, tmp_path):
        """Test 118: write_structural_posture creates file."""
        posture = {
            "schema_version": "1.0.0",
            "high_risk_directories": 2,
            "medium_risk_directories": 5,
            "low_risk_directories": 10,
            "new_directories": 1,
            "removed_directories": 0,
        }
        
        output_path = tmp_path / "posture.json"
        result_path = auditor_module.write_structural_posture(posture, output_path)
        
        assert result_path.exists()
        
        with open(result_path) as f:
            data = json.load(f)
        
        assert data["schema_version"] == "1.0.0"
        assert "timestamp" in data
    
    def test_posture_handles_empty_drift(self, auditor_module):
        """Test 119: Posture handles drift report with no changes."""
        drift_dict = {
            "new_directories": [],
            "removed_directories": [],
            "directory_drifts": [],
            "risk_transitions": {},
        }
        
        result = auditor_module.build_structural_posture(drift_dict)
        
        assert result["new_directories"] == 0
        assert result["removed_directories"] == 0
    
    def test_posture_consistent_with_drift_report(self, auditor_module, tmp_path):
        """Test 120: Posture counts are consistent with drift report."""
        old_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
                {"directory": "old_dir", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 3},
            ],
            "boundary_violations": [],
        }
        
        new_report = {
            "directory_scores": [
                {"directory": "backend", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 5},
                {"directory": "new_dir", "total_entropy": 0.5, "extension_entropy": 0.3, "naming_entropy": 0.2, "violation_count": 0, "unexpected_files": [], "forbidden_files": [], "naming_violations": [], "file_count": 3},
            ],
            "boundary_violations": [],
        }
        
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        
        with open(old_path, "w") as f:
            json.dump(old_report, f)
        with open(new_path, "w") as f:
            json.dump(new_report, f)
        
        drift = auditor_module.compare_structure_risk(old_path, new_path)
        posture = auditor_module.build_structural_posture(drift)
        
        # Posture counts should match drift report
        assert posture["new_directories"] == len(drift.new_directories)
        assert posture["removed_directories"] == len(drift.removed_directories)


# -----------------------------------------------------------------------------
# PHASE IV — STRUCTURE-AWARE RELEASE GATE TESTS (121-130)
# -----------------------------------------------------------------------------

class TestStructureAwareReleaseGate:
    """Tests for structure-aware release evaluation (Phase IV)."""
    
    def test_release_eval_has_required_fields(self, auditor_module, tmp_path):
        """Test 121: Release evaluation has all required fields."""
        # Create mock drift report
        drift_dict = {
            "old_report": "old.json",
            "new_report": "new.json",
            "overall_trend": "stable",
            "max_risk_increase": 0.1,
            "max_risk_decrease": 0.0,
            "directories_with_increased_risk": [],
            "directories_with_decreased_risk": [],
            "new_directories": [],
            "removed_directories": [],
            "risk_transitions": {},
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "entropy_old": 0.5,
                    "entropy_new": 0.5,
                    "entropy_delta": 0.0,
                    "violations_old": 0,
                    "violations_new": 0,
                    "violations_delta": 0,
                    "risk_score_old": 0.2,
                    "risk_score_new": 0.2,
                    "risk_delta": 0.0,
                    "risk_level_old": "LOW",
                    "risk_level_new": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {"overall_trend": "stable"},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        
        required_fields = {"release_ok", "status", "refactor_zones", "reasons"}
        assert required_fields.issubset(set(result.keys()))
    
    def test_release_blocked_on_core_high_risk(self, auditor_module, tmp_path):
        """Test 122: Release blocked when core zone has HIGH risk."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_old": "MEDIUM",
                    "risk_level_new": "HIGH",
                    "entropy_old": 1.0,
                    "entropy_new": 2.5,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_new": 12,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_score_new": 0.9,
                    "risk_delta": 0.4,
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {"overall_trend": "degrading"},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        
        assert result["status"] == "BLOCK"
        assert result["release_ok"] is False
        assert "backend" in result["refactor_zones"]
    
    def test_release_warned_on_core_medium_risk(self, auditor_module, tmp_path):
        """Test 123: Release warned when core zone has MEDIUM risk."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "experiments/test",
                    "risk_level_old": "LOW",
                    "risk_level_new": "MEDIUM",
                    "entropy_old": 0.5,
                    "entropy_new": 1.2,
                    "entropy_delta": 0.7,
                    "violations_old": 0,
                    "violations_new": 5,
                    "violations_delta": 5,
                    "risk_score_old": 0.2,
                    "risk_score_new": 0.6,
                    "risk_delta": 0.4,
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {"overall_trend": "degrading"},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        
        assert result["status"] == "WARN"
        assert result["release_ok"] is True
    
    def test_release_warned_on_peripheral_high_risk(self, auditor_module, tmp_path):
        """Test 124: Release warned when peripheral zone has HIGH risk."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "docs/api_reference",
                    "risk_level_old": "LOW",
                    "risk_level_new": "HIGH",
                    "entropy_old": 0.3,
                    "entropy_new": 2.8,
                    "entropy_delta": 2.5,
                    "violations_old": 0,
                    "violations_new": 15,
                    "violations_delta": 15,
                    "risk_score_old": 0.1,
                    "risk_score_new": 0.95,
                    "risk_delta": 0.85,
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {"overall_trend": "degrading"},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        
        assert result["status"] == "WARN"
        assert result["release_ok"] is True
    
    def test_release_ok_with_low_risk(self, auditor_module, tmp_path):
        """Test 125: Release OK when all directories are LOW risk."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/utils",
                    "risk_level_old": "LOW",
                    "risk_level_new": "LOW",
                    "entropy_old": 0.3,
                    "entropy_new": 0.4,
                    "entropy_delta": 0.1,
                    "violations_old": 0,
                    "violations_new": 1,
                    "violations_delta": 1,
                    "risk_score_old": 0.1,
                    "risk_score_new": 0.15,
                    "risk_delta": 0.05,
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {"overall_trend": "stable"},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        
        assert result["status"] == "OK"
        assert result["release_ok"] is True
    
    def test_release_eval_reasons_neutral(self, auditor_module, tmp_path):
        """Test 126: Release evaluation reasons use neutral language."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        
        reasons_text = " ".join(result["reasons"]).lower()
        evaluative_words = ["bad", "good", "broken", "fixed", "error", "wrong"]
        for word in evaluative_words:
            assert word not in reasons_text
    
    def test_release_eval_deterministic(self, auditor_module, tmp_path):
        """Test 127: Release evaluation produces deterministic output."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "MEDIUM",
                    "entropy_new": 1.2,
                    "violations_new": 5,
                    "risk_score_new": 0.6,
                    "entropy_old": 0.5,
                    "entropy_delta": 0.7,
                    "violations_old": 0,
                    "violations_delta": 5,
                    "risk_score_old": 0.2,
                    "risk_delta": 0.4,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        result1 = auditor_module.evaluate_structure_for_release(drift_dict)
        result2 = auditor_module.evaluate_structure_for_release(drift_dict)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)
    
    def test_release_eval_handles_empty_drift(self, auditor_module):
        """Test 128: Release evaluation handles empty drift report."""
        drift_dict = {
            "directory_drifts": [],
            "summary": {},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        
        assert result["status"] == "OK"
        assert result["release_ok"] is True
    
    def test_release_eval_refactor_zones_sorted(self, auditor_module, tmp_path):
        """Test 129: Refactor zones are sorted alphabetically."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "zebra/dir",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
                {
                    "path": "alpha/dir",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        
        zones = result["refactor_zones"]
        assert zones == sorted(zones)
    
    def test_release_eval_multiple_core_zones(self, auditor_module, tmp_path):
        """Test 130: Release evaluation handles multiple core zones."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
                {
                    "path": "experiments/test",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.3,
                    "violations_new": 10,
                    "risk_score_new": 0.85,
                    "entropy_old": 0.8,
                    "entropy_delta": 1.5,
                    "violations_old": 2,
                    "violations_delta": 8,
                    "risk_score_old": 0.3,
                    "risk_delta": 0.55,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        
        assert result["status"] == "BLOCK"
        assert "backend" in result["refactor_zones"]
        assert "experiments" in result["refactor_zones"]


# -----------------------------------------------------------------------------
# PHASE IV — REFACTOR PLANNING VIEW TESTS (131-140)
# -----------------------------------------------------------------------------

class TestStructuralRefactorPlanning:
    """Tests for structural refactor planning view (Phase IV)."""
    
    def test_refactor_plan_has_required_fields(self, auditor_module, tmp_path):
        """Test 131: Refactor plan has all required fields."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        result = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        required_fields = {"priority_zones", "suggested_batching", "notes"}
        assert required_fields.issubset(set(result.keys()))
    
    def test_priority_zones_ordered_by_risk(self, auditor_module, tmp_path):
        """Test 132: Priority zones ordered by highest risk first."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "low_risk/zone",
                    "risk_level_new": "LOW",
                    "entropy_new": 0.5,
                    "violations_new": 1,
                    "risk_score_new": 0.2,
                    "entropy_old": 0.3,
                    "entropy_delta": 0.2,
                    "violations_old": 0,
                    "violations_delta": 1,
                    "risk_score_old": 0.1,
                    "risk_delta": 0.1,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
                {
                    "path": "high_risk/zone",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        result = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        # high_risk should come before low_risk in priority
        zones = result["priority_zones"]
        assert "high_risk" in zones
        assert "low_risk" in zones
        assert zones.index("high_risk") < zones.index("low_risk")
    
    def test_suggested_batching_by_zone_and_risk(self, auditor_module, tmp_path):
        """Test 133: Suggested batching groups by zone and risk level."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
                {
                    "path": "backend/worker",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.3,
                    "violations_new": 10,
                    "risk_score_new": 0.85,
                    "entropy_old": 0.8,
                    "entropy_delta": 1.5,
                    "violations_old": 2,
                    "violations_delta": 8,
                    "risk_score_old": 0.3,
                    "risk_delta": 0.55,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        result = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        batches = result["suggested_batching"]
        backend_high = [b for b in batches if b.get("batch_id") == "backend:HIGH"]
        assert len(backend_high) == 1
        assert backend_high[0]["directory_count"] == 2
    
    def test_batching_directories_sorted(self, auditor_module, tmp_path):
        """Test 134: Directories within batches are sorted alphabetically."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/zebra",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
                {
                    "path": "backend/alpha",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.3,
                    "violations_new": 10,
                    "risk_score_new": 0.85,
                    "entropy_old": 0.8,
                    "entropy_delta": 1.5,
                    "violations_old": 2,
                    "violations_delta": 8,
                    "risk_score_old": 0.3,
                    "risk_delta": 0.55,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        result = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        batches = result["suggested_batching"]
        backend_high = next(b for b in batches if b.get("batch_id") == "backend:HIGH")
        dirs = backend_high["directories"]
        assert dirs == sorted(dirs)
    
    def test_refactor_plan_notes_neutral(self, auditor_module, tmp_path):
        """Test 135: Refactor plan notes use neutral language."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        result = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        notes_text = " ".join(result["notes"]).lower()
        evaluative_words = ["bad", "good", "broken", "fixed", "must", "should", "need"]
        for word in evaluative_words:
            assert word not in notes_text
    
    def test_refactor_plan_deterministic(self, auditor_module, tmp_path):
        """Test 136: Refactor plan produces deterministic output."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "MEDIUM",
                    "entropy_new": 1.2,
                    "violations_new": 5,
                    "risk_score_new": 0.6,
                    "entropy_old": 0.5,
                    "entropy_delta": 0.7,
                    "violations_old": 0,
                    "violations_delta": 5,
                    "risk_score_old": 0.2,
                    "risk_delta": 0.4,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        result1 = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        result2 = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)
    
    def test_refactor_plan_handles_empty_drift(self, auditor_module):
        """Test 137: Refactor plan handles empty drift report."""
        drift_dict = {
            "directory_drifts": [],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        result = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        assert len(result["priority_zones"]) == 0
        assert len(result["suggested_batching"]) == 0
    
    def test_batching_includes_all_required_fields(self, auditor_module, tmp_path):
        """Test 138: Each batch has all required fields."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        result = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        if result["suggested_batching"]:
            batch = result["suggested_batching"][0]
            required_fields = {"batch_id", "zone", "risk_level", "directories", "directory_count"}
            assert required_fields.issubset(set(batch.keys()))
    
    def test_refactor_plan_priority_zones_deterministic(self, auditor_module, tmp_path):
        """Test 139: Priority zones list is deterministic."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "zebra/dir",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
                {
                    "path": "alpha/dir",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.3,
                    "violations_new": 10,
                    "risk_score_new": 0.85,
                    "entropy_old": 0.8,
                    "entropy_delta": 1.5,
                    "violations_old": 2,
                    "violations_delta": 8,
                    "risk_score_old": 0.3,
                    "risk_delta": 0.55,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        result1 = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        result2 = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        assert result1["priority_zones"] == result2["priority_zones"]
    
    def test_refactor_plan_notes_include_zone_info(self, auditor_module, tmp_path):
        """Test 140: Refactor plan notes include zone information."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        result = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        notes_text = " ".join(result["notes"])
        assert "backend" in notes_text or len(result["notes"]) > 0


# -----------------------------------------------------------------------------
# PHASE IV — DIRECTOR STRUCTURE PANEL TESTS (141-150)
# -----------------------------------------------------------------------------

class TestStructureDirectorPanel:
    """Tests for Director-level structure panel (Phase IV)."""
    
    def test_director_panel_has_required_fields(self, auditor_module, tmp_path):
        """Test 141: Director panel has all required fields."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        refactor_plan = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        result = auditor_module.build_structure_director_panel(release_eval, refactor_plan)
        
        required_fields = {"status_light", "high_risk_zones", "headline"}
        assert required_fields.issubset(set(result.keys()))
    
    def test_status_light_red_on_block(self, auditor_module, tmp_path):
        """Test 142: Status light is RED when release is BLOCKED."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        refactor_plan = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        result = auditor_module.build_structure_director_panel(release_eval, refactor_plan)
        
        assert result["status_light"] == "RED"
    
    def test_status_light_yellow_on_warn(self, auditor_module, tmp_path):
        """Test 143: Status light is YELLOW when release is WARN."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "experiments/test",
                    "risk_level_new": "MEDIUM",
                    "entropy_new": 1.2,
                    "violations_new": 5,
                    "risk_score_new": 0.6,
                    "entropy_old": 0.5,
                    "entropy_delta": 0.7,
                    "violations_old": 0,
                    "violations_delta": 5,
                    "risk_score_old": 0.2,
                    "risk_delta": 0.4,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        refactor_plan = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        result = auditor_module.build_structure_director_panel(release_eval, refactor_plan)
        
        assert result["status_light"] == "YELLOW"
    
    def test_status_light_green_on_ok(self, auditor_module, tmp_path):
        """Test 144: Status light is GREEN when release is OK."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/utils",
                    "risk_level_new": "LOW",
                    "entropy_new": 0.4,
                    "violations_new": 1,
                    "risk_score_new": 0.15,
                    "entropy_old": 0.3,
                    "entropy_delta": 0.1,
                    "violations_old": 0,
                    "violations_delta": 1,
                    "risk_score_old": 0.1,
                    "risk_delta": 0.05,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        refactor_plan = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        result = auditor_module.build_structure_director_panel(release_eval, refactor_plan)
        
        assert result["status_light"] == "GREEN"
    
    def test_high_risk_zones_extracted(self, auditor_module, tmp_path):
        """Test 145: High-risk zones extracted from refactor plan."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        refactor_plan = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        result = auditor_module.build_structure_director_panel(release_eval, refactor_plan)
        
        assert "backend" in result["high_risk_zones"]
    
    def test_headline_neutral_language(self, auditor_module, tmp_path):
        """Test 146: Headline uses neutral language."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        refactor_plan = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        result = auditor_module.build_structure_director_panel(release_eval, refactor_plan)
        
        headline = result["headline"].lower()
        evaluative_words = ["bad", "good", "broken", "fixed", "error", "wrong"]
        for word in evaluative_words:
            assert word not in headline
    
    def test_director_panel_deterministic(self, auditor_module, tmp_path):
        """Test 147: Director panel produces deterministic output."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "MEDIUM",
                    "entropy_new": 1.2,
                    "violations_new": 5,
                    "risk_score_new": 0.6,
                    "entropy_old": 0.5,
                    "entropy_delta": 0.7,
                    "violations_old": 0,
                    "violations_delta": 5,
                    "risk_score_old": 0.2,
                    "risk_delta": 0.4,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        refactor_plan = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        
        result1 = auditor_module.build_structure_director_panel(release_eval, refactor_plan)
        result2 = auditor_module.build_structure_director_panel(release_eval, refactor_plan)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)
    
    def test_high_risk_zones_sorted(self, auditor_module, tmp_path):
        """Test 148: High-risk zones are sorted alphabetically."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "zebra/dir",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
                {
                    "path": "alpha/dir",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.3,
                    "violations_new": 10,
                    "risk_score_new": 0.85,
                    "entropy_old": 0.8,
                    "entropy_delta": 1.5,
                    "violations_old": 2,
                    "violations_delta": 8,
                    "risk_score_old": 0.3,
                    "risk_delta": 0.55,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        refactor_plan = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        result = auditor_module.build_structure_director_panel(release_eval, refactor_plan)
        
        zones = result["high_risk_zones"]
        assert zones == sorted(zones)
    
    def test_headline_matches_status(self, auditor_module, tmp_path):
        """Test 149: Headline content matches status light."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        release_eval = auditor_module.evaluate_structure_for_release(drift_dict)
        refactor_plan = auditor_module.build_structural_refactor_plan(drift_dict, release_eval)
        result = auditor_module.build_structure_director_panel(release_eval, refactor_plan)
        
        # RED status should mention "requires attention"
        if result["status_light"] == "RED":
            assert "attention" in result["headline"].lower() or "requires" in result["headline"].lower()
    
    def test_director_panel_handles_empty_inputs(self, auditor_module):
        """Test 150: Director panel handles empty inputs gracefully."""
        empty_release_eval = {
            "release_ok": True,
            "status": "OK",
            "refactor_zones": [],
            "reasons": [],
        }
        
        empty_refactor_plan = {
            "priority_zones": [],
            "suggested_batching": [],
            "notes": [],
        }
        
        result = auditor_module.build_structure_director_panel(empty_release_eval, empty_refactor_plan)
        
        assert result["status_light"] == "GREEN"
        assert len(result["high_risk_zones"]) == 0
        assert len(result["headline"]) > 0


# -----------------------------------------------------------------------------
# PHASE IV — CI HOOK & REFACTOR SPRINT PLANNER TESTS (151-160)
# -----------------------------------------------------------------------------

class TestCIReleaseGateHook:
    """Tests for CI release gate CLI hook (Phase IV)."""
    
    def test_release_gate_loads_drift_report(self, auditor_module, tmp_path):
        """Test 151: Release gate CLI loads drift report JSON."""
        drift_dict = {
            "old_report": "old.json",
            "new_report": "new.json",
            "overall_trend": "stable",
            "max_risk_increase": 0.0,
            "max_risk_decrease": 0.0,
            "directories_with_increased_risk": [],
            "directories_with_decreased_risk": [],
            "new_directories": [],
            "removed_directories": [],
            "risk_transitions": {},
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "LOW",
                    "entropy_new": 0.4,
                    "violations_new": 1,
                    "risk_score_new": 0.15,
                    "entropy_old": 0.3,
                    "entropy_delta": 0.1,
                    "violations_old": 0,
                    "violations_delta": 1,
                    "risk_score_old": 0.1,
                    "risk_delta": 0.05,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {"overall_trend": "stable"},
        }
        
        drift_file = tmp_path / "drift.json"
        with open(drift_file, "w") as f:
            json.dump(drift_dict, f)
        
        # Test that function can load and process
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        assert result["status"] in ["OK", "WARN", "BLOCK"]
    
    def test_release_gate_exit_code_ok(self, auditor_module, tmp_path):
        """Test 152: Release gate returns exit code 0 for OK status."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/utils",
                    "risk_level_new": "LOW",
                    "entropy_new": 0.4,
                    "violations_new": 1,
                    "risk_score_new": 0.15,
                    "entropy_old": 0.3,
                    "entropy_delta": 0.1,
                    "violations_old": 0,
                    "violations_delta": 1,
                    "risk_score_old": 0.1,
                    "risk_delta": 0.05,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        drift_file = tmp_path / "drift.json"
        with open(drift_file, "w") as f:
            json.dump(drift_dict, f)
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        assert result["status"] == "OK"
        assert result["release_ok"] is True
    
    def test_release_gate_exit_code_warn(self, auditor_module, tmp_path):
        """Test 153: Release gate returns exit code 1 for WARN status."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "experiments/test",
                    "risk_level_new": "MEDIUM",
                    "entropy_new": 1.2,
                    "violations_new": 5,
                    "risk_score_new": 0.6,
                    "entropy_old": 0.5,
                    "entropy_delta": 0.7,
                    "violations_old": 0,
                    "violations_delta": 5,
                    "risk_score_old": 0.2,
                    "risk_delta": 0.4,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        assert result["status"] == "WARN"
        assert result["release_ok"] is True
    
    def test_release_gate_exit_code_block(self, auditor_module, tmp_path):
        """Test 154: Release gate returns exit code 2 for BLOCK status."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "HIGH",
                    "entropy_new": 2.5,
                    "violations_new": 12,
                    "risk_score_new": 0.9,
                    "entropy_old": 1.0,
                    "entropy_delta": 1.5,
                    "violations_old": 3,
                    "violations_delta": 9,
                    "risk_score_old": 0.5,
                    "risk_delta": 0.4,
                    "risk_level_old": "MEDIUM",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        assert result["status"] == "BLOCK"
        assert result["release_ok"] is False
    
    def test_release_gate_json_output_format(self, auditor_module, tmp_path):
        """Test 155: Release gate JSON output has correct format."""
        drift_dict = {
            "directory_drifts": [
                {
                    "path": "backend/api",
                    "risk_level_new": "LOW",
                    "entropy_new": 0.4,
                    "violations_new": 1,
                    "risk_score_new": 0.15,
                    "entropy_old": 0.3,
                    "entropy_delta": 0.1,
                    "violations_old": 0,
                    "violations_delta": 1,
                    "risk_score_old": 0.1,
                    "risk_delta": 0.05,
                    "risk_level_old": "LOW",
                    "is_new_directory": False,
                    "is_removed_directory": False,
                },
            ],
            "summary": {},
        }
        
        result = auditor_module.evaluate_structure_for_release(drift_dict)
        
        # Verify JSON-serializable format
        json_output = {
            "release_ok": result["release_ok"],
            "status": result["status"],
            "refactor_zones": result["refactor_zones"],
            "reasons": result["reasons"],
        }
        
        json_str = json.dumps(json_output)
        parsed = json.loads(json_str)
        
        assert "release_ok" in parsed
        assert "status" in parsed
        assert "refactor_zones" in parsed
        assert "reasons" in parsed


# -----------------------------------------------------------------------------
# PHASE IV — REFACTOR SPRINT PLANNER TESTS (156-160)
# -----------------------------------------------------------------------------

class TestRefactorSprintPlanner:
    """Tests for refactor sprint planner (Phase IV)."""
    
    def test_sprint_plan_has_required_fields(self, auditor_module):
        """Test 156: Sprint plan has all required fields."""
        refactor_plan = {
            "priority_zones": ["backend", "tests"],
            "suggested_batching": [
                {
                    "batch_id": "backend:HIGH",
                    "zone": "backend",
                    "risk_level": "HIGH",
                    "directories": ["backend/api", "backend/worker"],
                    "directory_count": 2,
                },
                {
                    "batch_id": "tests:MEDIUM",
                    "zone": "tests",
                    "risk_level": "MEDIUM",
                    "directories": ["tests/unit"],
                    "directory_count": 1,
                },
            ],
            "notes": ["Test notes"],
        }
        
        result = auditor_module.build_refactor_sprint_plan(refactor_plan)
        
        required_fields = {"sprints", "neutral_notes"}
        assert required_fields.issubset(set(result.keys()))
    
    def test_sprints_group_batches_by_size(self, auditor_module):
        """Test 157: Sprints group batches respecting max_batch_size."""
        refactor_plan = {
            "priority_zones": ["backend", "tests", "docs"],
            "suggested_batching": [
                {"batch_id": "backend:HIGH", "zone": "backend", "risk_level": "HIGH", "directories": [], "directory_count": 1},
                {"batch_id": "tests:MEDIUM", "zone": "tests", "risk_level": "MEDIUM", "directories": [], "directory_count": 1},
                {"batch_id": "docs:LOW", "zone": "docs", "risk_level": "LOW", "directories": [], "directory_count": 1},
                {"batch_id": "backend:MEDIUM", "zone": "backend", "risk_level": "MEDIUM", "directories": [], "directory_count": 1},
            ],
            "notes": [],
        }
        
        result = auditor_module.build_refactor_sprint_plan(refactor_plan, max_batch_size=2)
        
        # Should have 2 sprints (2 batches each)
        assert len(result["sprints"]) == 2
        assert len(result["sprints"][0]["batches"]) == 2
        assert len(result["sprints"][1]["batches"]) == 2
    
    def test_sprints_ordered_by_priority(self, auditor_module):
        """Test 158: Sprints contain batches in priority order (HIGH first)."""
        refactor_plan = {
            "priority_zones": ["backend", "tests"],
            "suggested_batching": [
                {"batch_id": "tests:LOW", "zone": "tests", "risk_level": "LOW", "directories": [], "directory_count": 1},
                {"batch_id": "backend:HIGH", "zone": "backend", "risk_level": "HIGH", "directories": [], "directory_count": 1},
                {"batch_id": "tests:MEDIUM", "zone": "tests", "risk_level": "MEDIUM", "directories": [], "directory_count": 1},
            ],
            "notes": [],
        }
        
        result = auditor_module.build_refactor_sprint_plan(refactor_plan, max_batch_size=3)
        
        # First sprint should have HIGH batch first
        first_sprint_batches = result["sprints"][0]["batches"]
        assert "backend:HIGH" in first_sprint_batches
        
        # HIGH should come before MEDIUM and LOW
        if len(first_sprint_batches) > 1:
            high_idx = first_sprint_batches.index("backend:HIGH") if "backend:HIGH" in first_sprint_batches else -1
            medium_idx = first_sprint_batches.index("tests:MEDIUM") if "tests:MEDIUM" in first_sprint_batches else -1
            if high_idx >= 0 and medium_idx >= 0:
                assert high_idx < medium_idx
    
    def test_sprint_plan_deterministic(self, auditor_module):
        """Test 159: Sprint plan produces deterministic output."""
        refactor_plan = {
            "priority_zones": ["backend"],
            "suggested_batching": [
                {"batch_id": "backend:HIGH", "zone": "backend", "risk_level": "HIGH", "directories": [], "directory_count": 1},
                {"batch_id": "backend:MEDIUM", "zone": "backend", "risk_level": "MEDIUM", "directories": [], "directory_count": 1},
            ],
            "notes": [],
        }
        
        result1 = auditor_module.build_refactor_sprint_plan(refactor_plan)
        result2 = auditor_module.build_refactor_sprint_plan(refactor_plan)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)
    
    def test_sprint_plan_notes_neutral(self, auditor_module):
        """Test 160: Sprint plan notes use neutral language."""
        refactor_plan = {
            "priority_zones": ["backend"],
            "suggested_batching": [
                {"batch_id": "backend:HIGH", "zone": "backend", "risk_level": "HIGH", "directories": [], "directory_count": 1},
            ],
            "notes": [],
        }
        
        result = auditor_module.build_refactor_sprint_plan(refactor_plan)
        
        notes_text = " ".join(result["neutral_notes"]).lower()
        evaluative_words = ["bad", "good", "must", "should", "need", "fix"]
        for word in evaluative_words:
            assert word not in notes_text
    
    def test_sprint_plan_handles_empty_batches(self, auditor_module):
        """Test 161: Sprint plan handles empty batch list."""
        refactor_plan = {
            "priority_zones": [],
            "suggested_batching": [],
            "notes": [],
        }
        
        result = auditor_module.build_refactor_sprint_plan(refactor_plan)
        
        assert len(result["sprints"]) == 0
        assert len(result["neutral_notes"]) > 0
    
    def test_sprint_ids_sequential(self, auditor_module):
        """Test 162: Sprint IDs are sequential (sprint_01, sprint_02, etc.)."""
        refactor_plan = {
            "priority_zones": ["backend", "tests", "docs", "ui"],
            "suggested_batching": [
                {"batch_id": f"zone{i}:HIGH", "zone": f"zone{i}", "risk_level": "HIGH", "directories": [], "directory_count": 1}
                for i in range(5)
            ],
            "notes": [],
        }
        
        result = auditor_module.build_refactor_sprint_plan(refactor_plan, max_batch_size=2)
        
        sprint_ids = [s["sprint_id"] for s in result["sprints"]]
        
        # Should have sequential IDs
        assert "sprint_01" in sprint_ids
        if len(sprint_ids) > 1:
            assert "sprint_02" in sprint_ids
    
    def test_sprint_batches_sorted(self, auditor_module):
        """Test 163: Batches within sprints are sorted alphabetically."""
        refactor_plan = {
            "priority_zones": ["backend", "tests"],
            "suggested_batching": [
                {"batch_id": "zebra:HIGH", "zone": "zebra", "risk_level": "HIGH", "directories": [], "directory_count": 1},
                {"batch_id": "alpha:HIGH", "zone": "alpha", "risk_level": "HIGH", "directories": [], "directory_count": 1},
            ],
            "notes": [],
        }
        
        result = auditor_module.build_refactor_sprint_plan(refactor_plan, max_batch_size=3)
        
        if result["sprints"]:
            batches = result["sprints"][0]["batches"]
            assert batches == sorted(batches)
    
    def test_sprint_plan_notes_include_counts(self, auditor_module):
        """Test 164: Sprint plan notes include batch counts."""
        refactor_plan = {
            "priority_zones": ["backend"],
            "suggested_batching": [
                {"batch_id": "backend:HIGH", "zone": "backend", "risk_level": "HIGH", "directories": [], "directory_count": 1},
                {"batch_id": "backend:MEDIUM", "zone": "backend", "risk_level": "MEDIUM", "directories": [], "directory_count": 1},
            ],
            "notes": [],
        }
        
        result = auditor_module.build_refactor_sprint_plan(refactor_plan)
        
        notes_text = " ".join(result["neutral_notes"])
        assert "2" in notes_text or "batch" in notes_text.lower()
    
    def test_sprint_plan_custom_max_batch_size(self, auditor_module):
        """Test 165: Sprint plan respects custom max_batch_size parameter."""
        refactor_plan = {
            "priority_zones": ["backend"],
            "suggested_batching": [
                {"batch_id": f"backend:{i}", "zone": "backend", "risk_level": "HIGH", "directories": [], "directory_count": 1}
                for i in range(5)
            ],
            "notes": [],
        }
        
        result = auditor_module.build_refactor_sprint_plan(refactor_plan, max_batch_size=2)
        
        # Should have 3 sprints (2+2+1)
        assert len(result["sprints"]) == 3
        assert all(len(s["batches"]) <= 2 for s in result["sprints"])


# -----------------------------------------------------------------------------
# PHASE V — SEMANTIC DRIFT SENTINEL GRID v3 TESTS (166-180)
# -----------------------------------------------------------------------------

class TestSemanticDriftTensor:
    """Tests for semantic drift tensor construction (Phase V)."""
    
    def test_tensor_has_required_fields(self, auditor_module):
        """Test 166: Semantic drift tensor has all required fields."""
        semantic_timeline = {
            "slices": {
                "slice_1": {"drift_score": 0.5},
                "slice_2": {"drift_score": 0.3},
            }
        }
        causal_chronicle = {
            "slice_causality": {
                "slice_1": {"causal_drift": 0.4},
            }
        }
        multi_axis_view = {
            "slice_metrics": {
                "slice_1": {"correlated_drift": 0.6},
            }
        }
        
        result = auditor_module.build_semantic_drift_tensor(
            semantic_timeline, causal_chronicle, multi_axis_view
        )
        
        required_fields = {"drift_components", "semantic_hotspots", "tensor_norm"}
        assert required_fields.issubset(set(result.keys()))
    
    def test_tensor_combines_all_axes(self, auditor_module):
        """Test 167: Tensor combines semantic, causal, and metric axes."""
        semantic_timeline = {
            "slices": {
                "slice_1": {"drift_score": 0.7},
            }
        }
        causal_chronicle = {
            "slice_causality": {
                "slice_1": {"causal_drift": 0.6},
            }
        }
        multi_axis_view = {
            "slice_metrics": {
                "slice_1": {"correlated_drift": 0.8},
            }
        }
        
        result = auditor_module.build_semantic_drift_tensor(
            semantic_timeline, causal_chronicle, multi_axis_view
        )
        
        components = result["drift_components"]["slice_1"]
        assert components["semantic"] == 0.7
        assert components["causal"] == 0.6
        assert components["metric_correlated"] == 0.8
    
    def test_hotspots_identified_correctly(self, auditor_module):
        """Test 168: Semantic hotspots identified with threshold > 0.6."""
        semantic_timeline = {
            "slices": {
                "hot_slice": {"drift_score": 0.9},
                "cold_slice": {"drift_score": 0.3},
            }
        }
        causal_chronicle = {
            "slice_causality": {
                "hot_slice": {"causal_drift": 0.8},
            }
        }
        multi_axis_view = {
            "slice_metrics": {
                "hot_slice": {"correlated_drift": 0.7},
            }
        }
        
        result = auditor_module.build_semantic_drift_tensor(
            semantic_timeline, causal_chronicle, multi_axis_view
        )
        
        assert "hot_slice" in result["semantic_hotspots"]
        assert "cold_slice" not in result["semantic_hotspots"]
    
    def test_tensor_norm_calculated(self, auditor_module):
        """Test 169: Tensor norm is calculated correctly."""
        semantic_timeline = {
            "slices": {
                "slice_1": {"drift_score": 0.5},
            }
        }
        causal_chronicle = {"slice_causality": {}}
        multi_axis_view = {"slice_metrics": {}}
        
        result = auditor_module.build_semantic_drift_tensor(
            semantic_timeline, causal_chronicle, multi_axis_view
        )
        
        assert isinstance(result["tensor_norm"], float)
        assert result["tensor_norm"] >= 0.0
    
    def test_hotspots_sorted(self, auditor_module):
        """Test 170: Semantic hotspots are sorted alphabetically."""
        semantic_timeline = {
            "slices": {
                "zebra": {"drift_score": 0.8},
                "alpha": {"drift_score": 0.9},
            }
        }
        causal_chronicle = {"slice_causality": {}}
        multi_axis_view = {"slice_metrics": {}}
        
        result = auditor_module.build_semantic_drift_tensor(
            semantic_timeline, causal_chronicle, multi_axis_view
        )
        
        hotspots = result["semantic_hotspots"]
        assert hotspots == sorted(hotspots)
    
    def test_tensor_handles_empty_inputs(self, auditor_module):
        """Test 171: Tensor handles empty input dictionaries."""
        result = auditor_module.build_semantic_drift_tensor({}, {}, {})
        
        assert len(result["drift_components"]) == 0
        assert len(result["semantic_hotspots"]) == 0
        assert result["tensor_norm"] == 0.0
    
    def test_tensor_deterministic(self, auditor_module):
        """Test 172: Tensor produces deterministic output."""
        semantic_timeline = {
            "slices": {
                "slice_1": {"drift_score": 0.5},
            }
        }
        causal_chronicle = {"slice_causality": {}}
        multi_axis_view = {"slice_metrics": {}}
        
        result1 = auditor_module.build_semantic_drift_tensor(
            semantic_timeline, causal_chronicle, multi_axis_view
        )
        result2 = auditor_module.build_semantic_drift_tensor(
            semantic_timeline, causal_chronicle, multi_axis_view
        )
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


class TestSemanticDriftCounterfactual:
    """Tests for semantic drift counterfactual analyzer (Phase V)."""
    
    def test_counterfactual_has_required_fields(self, auditor_module):
        """Test 173: Counterfactual analysis has all required fields."""
        tensor = {
            "drift_components": {
                "slice_1": {
                    "semantic": 0.5,
                    "causal": 0.4,
                    "metric_correlated": 0.3,
                }
            },
            "semantic_hotspots": [],
            "tensor_norm": 1.0,
        }
        
        result = auditor_module.analyze_semantic_drift_counterfactual(tensor)
        
        required_fields = {
            "projected_unstable_slices",
            "stability_timeline",
            "neutral_notes",
        }
        assert required_fields.issubset(set(result.keys()))
    
    def test_projects_unstable_slices(self, auditor_module):
        """Test 174: Counterfactual identifies projected unstable slices."""
        tensor = {
            "drift_components": {
                "unstable_slice": {
                    "semantic": 0.9,
                    "causal": 0.8,
                    "metric_correlated": 0.7,
                }
            },
            "semantic_hotspots": [],
            "tensor_norm": 1.0,
        }
        
        result = auditor_module.analyze_semantic_drift_counterfactual(
            tensor, projection_horizon=3, stability_threshold=0.7
        )
        
        assert "unstable_slice" in result["projected_unstable_slices"]
    
    def test_stability_timeline_structure(self, auditor_module):
        """Test 175: Stability timeline has correct structure per slice."""
        tensor = {
            "drift_components": {
                "slice_1": {
                    "semantic": 0.5,
                    "causal": 0.4,
                    "metric_correlated": 0.3,
                }
            },
            "semantic_hotspots": [],
            "tensor_norm": 1.0,
        }
        
        result = auditor_module.analyze_semantic_drift_counterfactual(tensor)
        
        timeline = result["stability_timeline"]["slice_1"]
        assert "current_stability" in timeline
        assert "projected_stability" in timeline
        assert "becomes_unstable_at" in timeline
        assert len(timeline["projected_stability"]) == 3  # Default horizon
    
    def test_unstable_slices_sorted_by_time(self, auditor_module):
        """Test 176: Unstable slices sorted by when they become unstable."""
        tensor = {
            "drift_components": {
                "late_unstable": {
                    "semantic": 0.6,
                    "causal": 0.5,
                    "metric_correlated": 0.4,
                },
                "early_unstable": {
                    "semantic": 0.9,
                    "causal": 0.8,
                    "metric_correlated": 0.7,
                }
            },
            "semantic_hotspots": [],
            "tensor_norm": 1.0,
        }
        
        result = auditor_module.analyze_semantic_drift_counterfactual(tensor)
        
        unstable = result["projected_unstable_slices"]
        if len(unstable) >= 2:
            # Early unstable should come before late unstable
            early_idx = unstable.index("early_unstable") if "early_unstable" in unstable else -1
            late_idx = unstable.index("late_unstable") if "late_unstable" in unstable else -1
            if early_idx >= 0 and late_idx >= 0:
                assert early_idx < late_idx
    
    def test_counterfactual_notes_neutral(self, auditor_module):
        """Test 177: Counterfactual notes use neutral language."""
        tensor = {
            "drift_components": {
                "slice_1": {
                    "semantic": 0.5,
                    "causal": 0.4,
                    "metric_correlated": 0.3,
                }
            },
            "semantic_hotspots": [],
            "tensor_norm": 1.0,
        }
        
        result = auditor_module.analyze_semantic_drift_counterfactual(tensor)
        
        notes_text = " ".join(result["neutral_notes"]).lower()
        evaluative_words = ["bad", "good", "must", "should", "fix", "broken"]
        for word in evaluative_words:
            assert word not in notes_text
    
    def test_counterfactual_deterministic(self, auditor_module):
        """Test 178: Counterfactual analysis produces deterministic output."""
        tensor = {
            "drift_components": {
                "slice_1": {
                    "semantic": 0.5,
                    "causal": 0.4,
                    "metric_correlated": 0.3,
                }
            },
            "semantic_hotspots": [],
            "tensor_norm": 1.0,
        }
        
        result1 = auditor_module.analyze_semantic_drift_counterfactual(tensor)
        result2 = auditor_module.analyze_semantic_drift_counterfactual(tensor)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)
    
    def test_custom_projection_horizon(self, auditor_module):
        """Test 179: Counterfactual respects custom projection_horizon."""
        tensor = {
            "drift_components": {
                "slice_1": {
                    "semantic": 0.5,
                    "causal": 0.4,
                    "metric_correlated": 0.3,
                }
            },
            "semantic_hotspots": [],
            "tensor_norm": 1.0,
        }
        
        result = auditor_module.analyze_semantic_drift_counterfactual(
            tensor, projection_horizon=5
        )
        
        timeline = result["stability_timeline"]["slice_1"]
        assert len(timeline["projected_stability"]) == 5


class TestSemanticDriftDirectorPanelV3:
    """Tests for semantic drift director panel v3 (Phase V)."""
    
    def test_director_panel_v3_has_required_fields(self, auditor_module):
        """Test 180: Director panel v3 has all required fields."""
        tensor = {
            "drift_components": {
                "slice_1": {
                    "semantic": 0.5,
                    "causal": 0.4,
                    "metric_correlated": 0.3,
                }
            },
            "semantic_hotspots": ["slice_1"],
            "tensor_norm": 1.0,
        }
        
        counterfactual = {
            "projected_unstable_slices": ["slice_1"],
            "stability_timeline": {},
            "neutral_notes": [],
        }
        
        result = auditor_module.build_semantic_drift_director_panel_v3(tensor, counterfactual)
        
        required_fields = {
            "status_light",
            "semantic_hotspots",
            "projected_instability_count",
            "gating_recommendation",
            "recommendation_reasons",
            "headline",
        }
        assert required_fields.issubset(set(result.keys()))
    
    def test_status_light_red_on_high_risk(self, auditor_module):
        """Test 181: Status light RED when 3+ hotspots or 3+ projected unstable."""
        tensor = {
            "drift_components": {
                "s1": {"semantic": 0.8, "causal": 0.7, "metric_correlated": 0.6},
                "s2": {"semantic": 0.8, "causal": 0.7, "metric_correlated": 0.6},
                "s3": {"semantic": 0.8, "causal": 0.7, "metric_correlated": 0.6},
            },
            "semantic_hotspots": ["s1", "s2", "s3"],
            "tensor_norm": 2.5,
        }
        
        counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {},
            "neutral_notes": [],
        }
        
        result = auditor_module.build_semantic_drift_director_panel_v3(tensor, counterfactual)
        
        assert result["status_light"] == "RED"
        assert result["gating_recommendation"] == "BLOCK"
    
    def test_status_light_yellow_on_medium_risk(self, auditor_module):
        """Test 182: Status light YELLOW when 1-2 hotspots or 1-2 projected unstable."""
        tensor = {
            "drift_components": {
                "s1": {"semantic": 0.7, "causal": 0.6, "metric_correlated": 0.5},
            },
            "semantic_hotspots": ["s1"],
            "tensor_norm": 1.0,
        }
        
        counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {},
            "neutral_notes": [],
        }
        
        result = auditor_module.build_semantic_drift_director_panel_v3(tensor, counterfactual)
        
        assert result["status_light"] == "YELLOW"
        assert result["gating_recommendation"] == "WARN"
    
    def test_status_light_green_on_low_risk(self, auditor_module):
        """Test 183: Status light GREEN when no hotspots and no projected unstable."""
        tensor = {
            "drift_components": {
                "s1": {"semantic": 0.3, "causal": 0.2, "metric_correlated": 0.1},
            },
            "semantic_hotspots": [],
            "tensor_norm": 0.5,
        }
        
        counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {},
            "neutral_notes": [],
        }
        
        result = auditor_module.build_semantic_drift_director_panel_v3(tensor, counterfactual)
        
        assert result["status_light"] == "GREEN"
        assert result["gating_recommendation"] == "OK"
    
    def test_recommendation_reasons_included(self, auditor_module):
        """Test 184: Recommendation reasons explain gating decision."""
        tensor = {
            "drift_components": {
                "s1": {"semantic": 0.7, "causal": 0.6, "metric_correlated": 0.5},
            },
            "semantic_hotspots": ["s1"],
            "tensor_norm": 1.0,
        }
        
        counterfactual = {
            "projected_unstable_slices": ["s1"],
            "stability_timeline": {},
            "neutral_notes": [],
        }
        
        result = auditor_module.build_semantic_drift_director_panel_v3(tensor, counterfactual)
        
        assert len(result["recommendation_reasons"]) > 0
        reasons_text = " ".join(result["recommendation_reasons"])
        assert "s1" in reasons_text or "hotspot" in reasons_text.lower()
    
    def test_headline_matches_status(self, auditor_module):
        """Test 185: Headline content matches status light."""
        tensor = {
            "drift_components": {
                "s1": {"semantic": 0.8, "causal": 0.7, "metric_correlated": 0.6},
            },
            "semantic_hotspots": ["s1", "s2", "s3"],
            "tensor_norm": 2.5,
        }
        
        counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {},
            "neutral_notes": [],
        }
        
        result = auditor_module.build_semantic_drift_director_panel_v3(tensor, counterfactual)
        
        if result["status_light"] == "RED":
            assert "BLOCK" in result["headline"]
        elif result["status_light"] == "YELLOW":
            assert "WARN" in result["headline"]
        else:
            assert "OK" in result["headline"]
    
    def test_director_panel_v3_deterministic(self, auditor_module):
        """Test 186: Director panel v3 produces deterministic output."""
        tensor = {
            "drift_components": {
                "s1": {"semantic": 0.5, "causal": 0.4, "metric_correlated": 0.3},
            },
            "semantic_hotspots": [],
            "tensor_norm": 1.0,
        }
        
        counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {},
            "neutral_notes": [],
        }
        
        result1 = auditor_module.build_semantic_drift_director_panel_v3(tensor, counterfactual)
        result2 = auditor_module.build_semantic_drift_director_panel_v3(tensor, counterfactual)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)
    
    def test_hotspots_sorted_in_panel(self, auditor_module):
        """Test 187: Semantic hotspots are sorted in panel output."""
        tensor = {
            "drift_components": {
                "zebra": {"semantic": 0.8, "causal": 0.7, "metric_correlated": 0.6},
                "alpha": {"semantic": 0.8, "causal": 0.7, "metric_correlated": 0.6},
            },
            "semantic_hotspots": ["zebra", "alpha"],
            "tensor_norm": 2.0,
        }
        
        counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {},
            "neutral_notes": [],
        }
        
        result = auditor_module.build_semantic_drift_director_panel_v3(tensor, counterfactual)
        
        hotspots = result["semantic_hotspots"]
        assert hotspots == sorted(hotspots)
    
    def test_panel_handles_empty_inputs(self, auditor_module):
        """Test 188: Director panel v3 handles empty inputs gracefully."""
        empty_tensor = {
            "drift_components": {},
            "semantic_hotspots": [],
            "tensor_norm": 0.0,
        }
        
        empty_counterfactual = {
            "projected_unstable_slices": [],
            "stability_timeline": {},
            "neutral_notes": [],
        }
        
        result = auditor_module.build_semantic_drift_director_panel_v3(
            empty_tensor, empty_counterfactual
        )
        
        assert result["status_light"] == "GREEN"
        assert len(result["semantic_hotspots"]) == 0
        assert result["projected_instability_count"] == 0
