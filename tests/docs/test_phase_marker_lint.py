#!/usr/bin/env python3
"""
Tests for docs/phase_marker_lint.py - Phase marker consistency checker.

Tests:
- Phase marker detection
- Uplift disclaimer detection
- Forbidden claims detection
- Required sections checking
- Config-driven validation
"""

import sys
from pathlib import Path

import pytest
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from docs.phase_marker_lint import PhaseMarkerLinter


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure for testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_rules():
    """Sample rules for testing."""
    return {
        'rules': {
            'docs/PHASE2*.md': {
                'phase_marker': 'PHASE II',
                'require_uplift_disclaimer': True,
                'forbidden_claims': ['uplift has been demonstrated']
            }
        }
    }


def test_check_phase_marker_present(temp_project):
    """Test detection of phase marker."""
    linter = PhaseMarkerLinter(temp_project)
    
    content = """
# Phase II Planning

> **STATUS: PHASE II — NOT YET RUN**

This document describes Phase II experiments.
"""
    
    assert linter.check_phase_marker(content, "test.md", "PHASE II") is True


def test_check_phase_marker_missing(temp_project):
    """Test detection of missing phase marker."""
    linter = PhaseMarkerLinter(temp_project)
    
    content = """
# Some Documentation

This document does not have a phase marker.
"""
    
    assert linter.check_phase_marker(content, "test.md", "PHASE II") is False


def test_check_phase_marker_various_formats(temp_project):
    """Test detection of phase markers in various formats."""
    linter = PhaseMarkerLinter(temp_project)
    
    formats = [
        "**STATUS: PHASE II**",
        "> **STATUS: PHASE II — NOT YET RUN**",
        "# PHASE II - Planning Document",
        "PHASE II — Experimental Design",
    ]
    
    for fmt in formats:
        content = f"# Doc\n\n{fmt}\n\nContent here."
        assert linter.check_phase_marker(content, "test.md", "PHASE II") is True


def test_check_uplift_disclaimer_present(temp_project):
    """Test detection of uplift disclaimer."""
    linter = PhaseMarkerLinter(temp_project)
    
    content = """
# Uplift Experiments

We plan to test for uplift in Phase II.

**Note:** No empirical uplift has been demonstrated yet.
"""
    
    assert linter.check_uplift_disclaimer(content, "test.md") is True


def test_check_uplift_disclaimer_missing(temp_project):
    """Test detection of missing uplift disclaimer."""
    linter = PhaseMarkerLinter(temp_project)
    
    content = """
# Uplift Experiments

The RFL system demonstrates uplift over baseline.
"""
    
    assert linter.check_uplift_disclaimer(content, "test.md") is False


def test_check_uplift_disclaimer_not_needed(temp_project):
    """Test that disclaimer is not required if uplift not mentioned."""
    linter = PhaseMarkerLinter(temp_project)
    
    content = """
# System Architecture

This document describes the architecture.
No mention of experimental results here.
"""
    
    # Should return True because no uplift mention means no disclaimer needed
    assert linter.check_uplift_disclaimer(content, "test.md") is True


def test_check_uplift_disclaimer_various_formats(temp_project):
    """Test detection of disclaimers in various formats."""
    linter = PhaseMarkerLinter(temp_project)
    
    disclaimers = [
        "NO UPLIFT CLAIMS MAY BE MADE",
        "no empirical uplift yet",
        "no uplift has been demonstrated",
        "uplift has not been demonstrated",
        "NOT YET RUN",
        "NOT RUN IN PHASE I",
    ]
    
    for disclaimer in disclaimers:
        content = f"""
# Uplift Test

We test for uplift here.

{disclaimer}
"""
        assert linter.check_uplift_disclaimer(content, "test.md") is True


def test_check_forbidden_claims_found(temp_project):
    """Test detection of forbidden claims."""
    linter = PhaseMarkerLinter(temp_project)
    
    content = "Uplift has been demonstrated in our experiments."
    
    forbidden = ['uplift has been demonstrated', 'proven improvement']
    found = linter.check_forbidden_claims(content, "test.md", forbidden)
    
    assert len(found) == 1
    assert 'uplift has been demonstrated' in found


def test_check_forbidden_claims_not_found(temp_project):
    """Test when no forbidden claims are present."""
    linter = PhaseMarkerLinter(temp_project)
    
    content = "We plan to test for uplift in Phase II."
    
    forbidden = ['uplift has been demonstrated', 'proven improvement']
    found = linter.check_forbidden_claims(content, "test.md", forbidden)
    
    assert len(found) == 0


def test_check_required_sections_present(temp_project):
    """Test detection of required sections."""
    linter = PhaseMarkerLinter(temp_project)
    
    content = """
# Document Title

## Overview

Content here.

## Methods

More content.
"""
    
    required = ['Overview', 'Methods']
    missing = linter.check_required_sections(content, "test.md", required)
    
    assert len(missing) == 0


def test_check_required_sections_missing(temp_project):
    """Test detection of missing required sections."""
    linter = PhaseMarkerLinter(temp_project)
    
    content = """
# Document Title

## Overview

Content here.
"""
    
    required = ['Overview', 'Methods', 'Results']
    missing = linter.check_required_sections(content, "test.md", required)
    
    assert len(missing) == 2
    assert 'Methods' in missing
    assert 'Results' in missing


def test_check_file_all_pass(temp_project):
    """Test checking a file that passes all rules."""
    doc_path = temp_project / "docs" / "PHASE2_TEST.md"
    doc_path.parent.mkdir(exist_ok=True)
    
    doc_path.write_text("""
# Phase II Test Plan

> **STATUS: PHASE II — NOT YET RUN**

## Overview

We plan to test for uplift in Phase II.

**Note:** No empirical uplift has been demonstrated yet.
""")
    
    rules = {
        'phase_marker': 'PHASE II',
        'require_uplift_disclaimer': True,
        'required_sections': ['Overview']
    }
    
    linter = PhaseMarkerLinter(temp_project)
    result = linter.check_file(doc_path, rules)
    
    assert result is True
    assert len(linter.errors) == 0


def test_check_file_missing_phase_marker(temp_project):
    """Test checking a file missing required phase marker."""
    doc_path = temp_project / "docs" / "PHASE2_TEST.md"
    doc_path.parent.mkdir(exist_ok=True)
    
    doc_path.write_text("""
# Test Plan

This document is missing a phase marker.
""")
    
    rules = {
        'phase_marker': 'PHASE II'
    }
    
    linter = PhaseMarkerLinter(temp_project)
    result = linter.check_file(doc_path, rules)
    
    assert result is False
    assert len(linter.errors) == 1
    assert 'PHASE II' in linter.errors[0]


def test_check_file_missing_disclaimer(temp_project):
    """Test checking a file missing uplift disclaimer."""
    doc_path = temp_project / "docs" / "PHASE2_TEST.md"
    doc_path.parent.mkdir(exist_ok=True)
    
    doc_path.write_text("""
# Uplift Results

> **STATUS: PHASE II**

The system demonstrates uplift over baseline.
""")
    
    rules = {
        'phase_marker': 'PHASE II',
        'require_uplift_disclaimer': True
    }
    
    linter = PhaseMarkerLinter(temp_project)
    result = linter.check_file(doc_path, rules)
    
    assert result is False
    assert len(linter.errors) == 1
    assert 'disclaimer' in linter.errors[0].lower()


def test_check_file_forbidden_claim(temp_project):
    """Test detection of forbidden claim in file."""
    doc_path = temp_project / "docs" / "TEST.md"
    doc_path.parent.mkdir(exist_ok=True)
    
    doc_path.write_text("""
# Results

Uplift has been demonstrated in Phase I.
""")
    
    rules = {
        'forbidden_claims': ['uplift has been demonstrated']
    }
    
    linter = PhaseMarkerLinter(temp_project)
    result = linter.check_file(doc_path, rules)
    
    assert result is False
    assert len(linter.errors) == 1
    assert 'forbidden claim' in linter.errors[0].lower()


def test_lint_with_config(temp_project):
    """Test full linting workflow with config."""
    # Create test document
    doc_path = temp_project / "docs" / "PHASE2_PLAN.md"
    doc_path.parent.mkdir(exist_ok=True)
    
    doc_path.write_text("""
# Phase II Planning

> **STATUS: PHASE II — NOT YET RUN**

## Overview

We will test for uplift.

**Note:** No uplift has been demonstrated yet.
""")
    
    # Create config
    config = {
        'rules': {
            'docs/PHASE2*.md': {
                'phase_marker': 'PHASE II',
                'require_uplift_disclaimer': True,
                'required_sections': ['Overview']
            }
        }
    }
    
    linter = PhaseMarkerLinter(temp_project)
    exit_code = linter.lint(config)
    
    assert exit_code == 0
    assert len(linter.errors) == 0


def test_lint_with_errors(temp_project):
    """Test linting that finds errors."""
    # Create test document with issues
    doc_path = temp_project / "docs" / "PHASE2_PLAN.md"
    doc_path.parent.mkdir(exist_ok=True)
    
    doc_path.write_text("""
# Phase II Planning

We found that uplift has been demonstrated.
""")
    
    # Create config
    config = {
        'rules': {
            'docs/PHASE2*.md': {
                'phase_marker': 'PHASE II',
                'require_uplift_disclaimer': True,
                'forbidden_claims': ['uplift has been demonstrated']
            }
        }
    }
    
    linter = PhaseMarkerLinter(temp_project)
    exit_code = linter.lint(config)
    
    assert exit_code == 1
    assert len(linter.errors) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
