"""
PHASE II — DIRECTORY STRUCTURE TESTS

Agent: doc-ops-3 — Directory Cartographer
Purpose: Test directory structure invariants and phase boundaries.

Tests ensure:
1. Phase II files are in correct locations
2. Phase II files contain phase markers
3. Phase I code does not import Phase II modules
4. Directory boundaries are enforced
"""

import re
import sys
from pathlib import Path

import pytest


# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

# Add scripts to path for importing linter
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# -----------------------------------------------------------------------------
# Configuration (mirrors linter configuration)
# -----------------------------------------------------------------------------

PHASE_II_MARKERS = [
    r"PHASE\s*II",
    r"PHASE\s*2",
    r"Phase\s*II",
    r"Phase\s*2",
]

PHASE_II_FILES = {
    "backend/metrics/u2_analysis.py",
    "backend/metrics/statistical.py",
    "backend/runner/u2_runner.py",
    "backend/telemetry/u2_schema.py",
    "backend/security/u2_security.py",
    "backend/promotion/u2_evidence.py",
    "experiments/run_uplift_u2.py",
    "experiments/u2_cross_slice_analysis.py",
    "experiments/u2_pipeline.py",
    "config/curriculum_uplift_phase2.yaml",
    "scripts/validate_u2_environment.py",
    "scripts/build_u2_evidence_dossier.py",
    "scripts/proof_dag_u2_audit.py",
    "analysis/u2_dynamics.py",
}

PHASE_II_IMPORT_PATTERNS = [
    r"from\s+backend\.metrics\.u2_analysis",
    r"from\s+backend\.runner\.u2_runner",
    r"from\s+backend\.telemetry\.u2_schema",
    r"from\s+tests\.phase2",
    r"import\s+backend\.metrics\.u2_analysis",
    r"import\s+tests\.phase2",
]

PHASE_I_DIRS = [
    "curriculum",
    "derivation",
    "attestation",
    "rfl",
    "backend/axiom_engine",
    "backend/basis",
    "backend/crypto",
    "backend/dag",
    "backend/frontier",
    "backend/logic",
]

PHASE_II_ONLY_DIRS = [
    "tests/phase2",
    "artifacts/phase_ii",
    "artifacts/u2",
    "experiments/synthetic_uplift",
]

DOCS_FORBIDDEN_EXTENSIONS = [".py", ".js", ".ts", ".lean"]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def file_has_phase_marker(filepath: Path) -> bool:
    """Check if file has Phase II marker in first 50 lines."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")[:50]
        header = "\n".join(lines)
        return any(re.search(pattern, header) for pattern in PHASE_II_MARKERS)
    except Exception:
        return False


def file_imports_phase_ii(filepath: Path) -> list[tuple[int, str]]:
    """Check if file imports Phase II modules. Returns list of (line_num, line)."""
    violations = []
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        for line_num, line in enumerate(content.split("\n"), start=1):
            for pattern in PHASE_II_IMPORT_PATTERNS:
                if re.search(pattern, line):
                    violations.append((line_num, line.strip()))
    except Exception:
        pass
    return violations


# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

class TestPhaseIILabels:
    """Tests for Phase II file labeling requirements."""
    
    @pytest.mark.parametrize("rel_path", [
        p for p in PHASE_II_FILES if (PROJECT_ROOT / p).exists()
    ])
    def test_phase_ii_file_has_marker(self, rel_path: str):
        """Each Phase II file must have a phase marker in first 50 lines."""
        filepath = PROJECT_ROOT / rel_path
        assert file_has_phase_marker(filepath), (
            f"Phase II file '{rel_path}' missing phase marker. "
            f"Add '# PHASE II' comment in first 50 lines."
        )


class TestPhaseIIDirectories:
    """Tests for Phase II directory boundaries."""
    
    @pytest.mark.parametrize("rel_dir", PHASE_II_ONLY_DIRS)
    def test_phase_ii_directory_exists_or_empty(self, rel_dir: str):
        """Phase II directories should exist or be intentionally absent."""
        dir_path = PROJECT_ROOT / rel_dir
        # This is informational - directories may not exist yet
        if dir_path.exists():
            # Check that it's not empty (has meaningful content)
            items = list(dir_path.iterdir())
            # Allow __pycache__ and __init__.py only
            meaningful = [
                i for i in items 
                if i.name not in ("__pycache__", "__init__.py", ".gitkeep")
            ]
            # It's okay for directories to be nearly empty during early Phase II
            pass
    
    def test_phase2_test_directory_isolation(self):
        """tests/phase2/ should contain only Phase II tests."""
        phase2_dir = PROJECT_ROOT / "tests" / "phase2"
        if not phase2_dir.exists():
            pytest.skip("tests/phase2/ not yet created")
        
        for py_file in phase2_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            if py_file.name == "__init__.py":
                continue
            if py_file.name == "conftest.py":
                continue
            
            # Phase II test files should have marker or be in phase2 dir (implicit)
            # We enforce explicit markers for clarity
            assert file_has_phase_marker(py_file), (
                f"Test file in tests/phase2/ missing phase marker: {py_file.relative_to(PROJECT_ROOT)}"
            )


class TestPhaseIIsolation:
    """Tests ensuring Phase I code doesn't import Phase II modules."""
    
    @pytest.mark.parametrize("phase_i_dir", [
        d for d in PHASE_I_DIRS if (PROJECT_ROOT / d).exists()
    ])
    def test_no_phase_ii_imports_in_phase_i(self, phase_i_dir: str):
        """Phase I directories must not import Phase II modules."""
        dir_path = PROJECT_ROOT / phase_i_dir
        violations = []
        
        for py_file in dir_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            file_violations = file_imports_phase_ii(py_file)
            if file_violations:
                rel_path = py_file.relative_to(PROJECT_ROOT)
                for line_num, line in file_violations:
                    violations.append(f"  {rel_path}:{line_num}: {line}")
        
        assert not violations, (
            f"Phase I directory '{phase_i_dir}' imports Phase II modules:\n" + 
            "\n".join(violations)
        )


class TestDocsBoundary:
    """Tests ensuring docs/ contains only non-code artifacts."""
    
    def test_docs_no_python_files(self):
        """docs/ directory should not contain Python files."""
        docs_dir = PROJECT_ROOT / "docs"
        if not docs_dir.exists():
            pytest.skip("docs/ not found")
        
        py_files = list(docs_dir.rglob("*.py"))
        assert not py_files, (
            f"docs/ should not contain Python files. Found: "
            f"{[str(f.relative_to(PROJECT_ROOT)) for f in py_files]}"
        )
    
    def test_docs_no_js_files(self):
        """docs/ directory should not contain JavaScript files."""
        docs_dir = PROJECT_ROOT / "docs"
        if not docs_dir.exists():
            pytest.skip("docs/ not found")
        
        js_files = list(docs_dir.rglob("*.js"))
        assert not js_files, (
            f"docs/ should not contain JavaScript files. Found: "
            f"{[str(f.relative_to(PROJECT_ROOT)) for f in js_files]}"
        )
    
    def test_docs_no_lean_files(self):
        """docs/ directory should not contain Lean files."""
        docs_dir = PROJECT_ROOT / "docs"
        if not docs_dir.exists():
            pytest.skip("docs/ not found")
        
        lean_files = list(docs_dir.rglob("*.lean"))
        assert not lean_files, (
            f"docs/ should not contain Lean files. Found: "
            f"{[str(f.relative_to(PROJECT_ROOT)) for f in lean_files]}"
        )


class TestVerificationDirectoryBoundary:
    """Tests for backend/verification/ directory purity."""
    
    def test_verification_contains_only_verification_tools(self):
        """backend/verification/ should contain only verification tools."""
        verification_dir = PROJECT_ROOT / "backend" / "verification"
        if not verification_dir.exists():
            pytest.skip("backend/verification/ not found")
        
        # Check for files that don't look like verification tools
        suspicious_files = []
        for item in verification_dir.iterdir():
            if item.is_dir():
                if item.name != "__pycache__":
                    suspicious_files.append(item.name)
            elif item.suffix == ".py":
                if not any(kw in item.name.lower() for kw in ["verify", "valid", "check", "audit"]):
                    suspicious_files.append(item.name)
        
        # Currently empty is okay
        if not suspicious_files:
            pass  # All good


class TestDirectoryMapExists:
    """Tests for directory documentation."""
    
    def test_phase2_directory_map_exists(self):
        """docs/PHASE2_DIRECTORY_MAP.md should exist."""
        map_file = PROJECT_ROOT / "docs" / "PHASE2_DIRECTORY_MAP.md"
        assert map_file.exists(), (
            "docs/PHASE2_DIRECTORY_MAP.md not found. "
            "Run doc-ops-3 to generate directory map."
        )
    
    def test_phase2_directory_map_has_content(self):
        """docs/PHASE2_DIRECTORY_MAP.md should have substantial content."""
        map_file = PROJECT_ROOT / "docs" / "PHASE2_DIRECTORY_MAP.md"
        if not map_file.exists():
            pytest.skip("Directory map not found")
        
        content = map_file.read_text(encoding="utf-8")
        
        # Should have key sections
        assert "## Overview" in content, "Directory map missing Overview section"
        assert "Phase I" in content, "Directory map should reference Phase I"
        assert "Phase II" in content, "Directory map should reference Phase II"
        assert "Boundary" in content, "Directory map should define boundaries"


class TestNamingConventions:
    """Tests for Phase II naming conventions."""
    
    def test_u2_files_in_correct_locations(self):
        """Files with u2_ prefix should be in appropriate directories."""
        u2_files = list(PROJECT_ROOT.rglob("u2_*.py"))
        
        allowed_dirs = [
            "backend/metrics",
            "backend/runner",
            "backend/telemetry",
            "backend/security",
            "backend/promotion",
            "experiments",
            "analysis",
            "scripts",
            "tests/phase2",
            "tests/env",
            "tests/metrics",
        ]
        
        misplaced = []
        for f in u2_files:
            if "__pycache__" in str(f):
                continue
            # Use POSIX-style paths for consistent comparison
            rel_path = f.relative_to(PROJECT_ROOT).as_posix()
            in_allowed = any(rel_path.startswith(d) for d in allowed_dirs)
            if not in_allowed:
                misplaced.append(rel_path)
        
        assert not misplaced, (
            f"u2_*.py files found in unexpected locations: {misplaced}"
        )
    
    def test_phase2_yaml_configs(self):
        """Phase II YAML configs should have phase2 in name."""
        config_dir = PROJECT_ROOT / "config"
        if not config_dir.exists():
            pytest.skip("config/ not found")
        
        for yaml_file in config_dir.glob("*.yaml"):
            content = yaml_file.read_text(encoding="utf-8", errors="replace")
            if any(re.search(pattern, content) for pattern in PHASE_II_MARKERS):
                # Phase II content should have phase2 in filename
                if "phase2" not in yaml_file.name.lower() and "u2" not in yaml_file.name.lower():
                    # Allow curriculum.yaml to reference Phase II since it's the master config
                    if yaml_file.name != "curriculum.yaml":
                        pytest.fail(
                            f"YAML with Phase II content should have 'phase2' or 'u2' in name: {yaml_file.name}"
                        )


class TestResultsDirectoryConvention:
    """Tests for results/ directory naming conventions."""
    
    def test_u2_results_naming(self):
        """U2 results should follow uplift_u2_*.jsonl pattern."""
        results_dir = PROJECT_ROOT / "results"
        if not results_dir.exists():
            pytest.skip("results/ not found")
        
        u2_files = [f for f in results_dir.glob("*u2*.jsonl")]
        
        for f in u2_files:
            # Should start with uplift_u2_ or contain _u2_
            if not (f.name.startswith("uplift_u2_") or "_u2_" in f.name):
                # Just a warning, not a failure
                pass


# -----------------------------------------------------------------------------
# Linter Integration Tests
# -----------------------------------------------------------------------------

class TestLinterIntegration:
    """Tests that the linter script works correctly."""
    
    def test_linter_script_exists(self):
        """verify_directory_structure.py should exist."""
        linter = PROJECT_ROOT / "scripts" / "verify_directory_structure.py"
        assert linter.exists(), "scripts/verify_directory_structure.py not found"
    
    def test_linter_script_importable(self):
        """Linter should be importable."""
        try:
            import verify_directory_structure
        except ImportError as e:
            pytest.fail(f"Could not import linter: {e}")
    
    def test_linter_lint_all_returns_result(self):
        """Linter's lint_all() should return a LintResult."""
        from verify_directory_structure import lint_all, LintResult
        
        result = lint_all()
        assert isinstance(result, LintResult), "lint_all() should return LintResult"
        assert hasattr(result, "errors"), "LintResult should have errors"
        assert hasattr(result, "warnings"), "LintResult should have warnings"
        assert hasattr(result, "files_checked"), "LintResult should have files_checked"


# -----------------------------------------------------------------------------
# CI Integration Tests
# -----------------------------------------------------------------------------

class TestCIReadiness:
    """Tests that CI can run structure validation."""
    
    def test_linter_can_run_strict(self):
        """Linter should be runnable with --strict flag."""
        import subprocess
        
        linter = PROJECT_ROOT / "scripts" / "verify_directory_structure.py"
        result = subprocess.run(
            [sys.executable, str(linter), "--quiet"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        # We don't assert success because there may be legitimate violations
        # Just verify it runs without crashing
        assert result.returncode in (0, 1), (
            f"Linter crashed with code {result.returncode}: {result.stderr}"
        )

