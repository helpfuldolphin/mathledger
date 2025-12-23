"""
Tests for Doc Governance Radar

Validates detection of:
- Premature uplift claims
- TDA enforcement claims without wiring
- Phase X language violations (P3/P4)
- Substrate alignment claims
"""

import pytest
import json
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "radars"))

from doc_governance_radar import DocGovernanceRadar, EXIT_PASS, EXIT_FAIL, EXIT_WARN, EXIT_SKIP


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary repository structure."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    system_law_dir = docs_dir / "system_law"
    system_law_dir.mkdir()
    
    output_dir = tmp_path / "artifacts" / "drift"
    output_dir.mkdir(parents=True)
    
    return {
        "root": tmp_path,
        "docs": docs_dir,
        "system_law": system_law_dir,
        "output": output_dir
    }


def test_uplift_claim_without_evidence_fails(temp_repo):
    """Test that uplift claim without evidence is detected as FAIL."""
    doc_path = temp_repo["docs"] / "PHASE2_RFL_UPLIFT_PLAN.md"
    doc_path.write_text("""
# Phase 2 Plan

Our experiments proved uplift in the RFL system.
The results show that uplift achieved across all metrics.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_FAIL
    assert radar.drift_report["summary"]["critical"] >= 2
    assert any(v["type"] == "premature_uplift_claim" for v in radar.drift_report["violations"])


def test_uplift_claim_with_evidence_passes(temp_repo):
    """Test that uplift claim with proper evidence citation passes."""
    doc_path = temp_repo["docs"] / "PHASE2_RFL_UPLIFT_PLAN.md"
    doc_path.write_text("""
# Phase 2 Plan

Our experiments demonstrated uplift (see P3/P4 evidence package).
Results pending: integrated-run pending for final validation.
All G1-G5 gates passed with uplift confirmed.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_PASS
    assert radar.drift_report["summary"]["critical"] == 0


def test_uplift_claim_with_integrated_run_pending_passes(temp_repo):
    """Test that uplift claim with 'integrated-run pending' qualifier passes."""
    doc_path = temp_repo["docs"] / "PHASE2_RFL_UPLIFT_PLAN.md"
    doc_path.write_text("""
# Phase 2 Results

The system proved uplift in preliminary runs.
Final validation: integrated-run pending.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_PASS
    assert radar.drift_report["summary"]["critical"] == 0


def test_p3_described_as_real_world_fails(temp_repo):
    """Test that P3 described as 'real world' is detected as FAIL."""
    doc_path = temp_repo["system_law"] / "Phase_X_Prelaunch_Review.md"
    doc_path.write_text("""
# Phase X Prelaunch

P3 will run in real world conditions.
P3 is our production environment.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_FAIL
    assert radar.drift_report["summary"]["critical"] >= 1
    assert any(v["type"] == "p3_language_violation" for v in radar.drift_report["violations"])


def test_p3_described_as_synthetic_wind_tunnel_passes(temp_repo):
    """Test that P3 described as 'synthetic wind tunnel' passes."""
    doc_path = temp_repo["system_law"] / "Phase_X_Prelaunch_Review.md"
    doc_path.write_text("""
# Phase X Prelaunch

P3 is a synthetic wind tunnel environment.
P3 simulates conditions in a controlled setting.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_PASS
    assert radar.drift_report["summary"]["critical"] == 0


def test_tda_enforcement_live_without_qualifier_fails(temp_repo):
    """Test that TDA enforcement claim without wiring is detected as FAIL."""
    doc_path = temp_repo["docs"] / "TDA_MODES.md"
    doc_path.write_text("""
# TDA Modes

TDA enforcement live as of today.
TDA is now the final arbiter of all decisions.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_FAIL
    assert radar.drift_report["summary"]["critical"] >= 1
    assert any(v["type"] == "premature_tda_claim" for v in radar.drift_report["violations"])


def test_tda_enforcement_with_qualifier_passes(temp_repo):
    """Test that TDA enforcement claim with proper qualifier passes."""
    doc_path = temp_repo["docs"] / "TDA_MODES.md"
    doc_path.write_text("""
# TDA Modes

TDA enforcement (not yet wired) will be implemented.
TDA is now the final arbiter design pending integration.
TDA hooks planned for next phase.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_PASS
    assert radar.drift_report["summary"]["critical"] == 0


def test_substrate_solves_alignment_fails(temp_repo):
    """Test that claiming Substrate solves alignment is detected as FAIL."""
    doc_path = temp_repo["docs"] / "CORTEX_INTEGRATION.md"
    doc_path.write_text("""
# Cortex Integration

Our Substrate solves alignment completely.
With this system, alignment is guaranteed.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_FAIL
    assert radar.drift_report["summary"]["critical"] >= 1
    assert any(v["type"] == "substrate_alignment_claim" for v in radar.drift_report["violations"])


def test_p4_control_authority_without_shadow_qualifier_fails(temp_repo):
    """Test that P4 control claims without 'shadow' qualifier fail."""
    doc_path = temp_repo["system_law"] / "Phase_X_P3P4_TODO.md"
    doc_path.write_text("""
# P4 Implementation

P4 has control over the system.
P4 controls all critical paths.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_FAIL
    assert radar.drift_report["summary"]["critical"] >= 1
    assert any(v["type"] == "p4_language_violation" for v in radar.drift_report["violations"])


def test_p4_shadow_mode_passes(temp_repo):
    """Test that P4 described as shadow mode passes."""
    doc_path = temp_repo["system_law"] / "Phase_X_P3P4_TODO.md"
    doc_path.write_text("""
# P4 Implementation

P4 runs in shadow mode with no control authority.
P4 has no control over production systems.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_PASS
    assert radar.drift_report["summary"]["critical"] == 0


def test_no_documents_returns_skip(temp_repo):
    """Test that missing all documents returns SKIP."""
    # Don't create any documents
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_SKIP
    assert radar.drift_report["status"] == "SKIP"


def test_multiple_violations_in_same_document(temp_repo):
    """Test that multiple violations in the same document are all detected."""
    doc_path = temp_repo["docs"] / "PHASE2_RFL_UPLIFT_PLAN.md"
    doc_path.write_text("""
# Phase 2 Results

We proved uplift in our experiments.
P3 will run in real world production.
TDA enforcement live starting today.
The Substrate solves alignment.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_FAIL
    assert radar.drift_report["summary"]["critical"] >= 4
    
    violation_types = {v["type"] for v in radar.drift_report["violations"]}
    assert "premature_uplift_claim" in violation_types
    assert "p3_language_violation" in violation_types
    assert "premature_tda_claim" in violation_types
    assert "substrate_alignment_claim" in violation_types


def test_report_artifacts_created(temp_repo):
    """Test that JSON report and markdown summary are created."""
    doc_path = temp_repo["docs"] / "PHASE2_RFL_UPLIFT_PLAN.md"
    doc_path.write_text("# Clean document with no violations")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    radar.run()
    
    report_path = temp_repo["output"] / "doc_governance_report.json"
    summary_path = temp_repo["output"] / "doc_governance_summary.md"
    
    assert report_path.exists()
    assert summary_path.exists()
    
    # Verify JSON is valid
    with open(report_path) as f:
        report = json.load(f)
        assert "version" in report
        assert "status" in report
        assert "violations" in report


def test_line_numbers_reported_correctly(temp_repo):
    """Test that line numbers in violations are accurate."""
    doc_path = temp_repo["docs"] / "PHASE2_RFL_UPLIFT_PLAN.md"
    doc_path.write_text("Line 1\nLine 2\nLine 3\nWe proved uplift here on line 4.\nLine 5")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    radar.run()
    
    violations = [v for v in radar.drift_report["violations"] 
                  if v["type"] == "premature_uplift_claim"]
    assert len(violations) > 0
    assert violations[0]["line"] == 4


def test_case_insensitive_detection(temp_repo):
    """Test that detection is case-insensitive."""
    doc_path = temp_repo["docs"] / "PHASE2_RFL_UPLIFT_PLAN.md"
    doc_path.write_text("""
We PROVED UPLIFT in tests.
Uplift Achieved across metrics.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_FAIL
    assert radar.drift_report["summary"]["critical"] >= 2


def test_readme_checked_by_default(temp_repo):
    """Test that README.md is checked by default."""
    readme_path = temp_repo["root"] / "README.md"
    readme_path.write_text("""
# MathLedger

We proved uplift in our system.
""")
    
    radar = DocGovernanceRadar(temp_repo["root"], temp_repo["output"])
    exit_code = radar.run()
    
    assert exit_code == EXIT_FAIL
    violations = [v for v in radar.drift_report["violations"] 
                  if v["filepath"] == "README.md"]
    assert len(violations) > 0
