#!/usr/bin/env python3
"""
Tests for Documentation Drift Radar

Tests cover:
- Document fingerprinting
- Drift detection (missing markers, uplifting language, code drift, terminology drift)
- Governance validation
- CLI argument handling
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from docs.docs_fingerprint import (
    normalize_text,
    count_code_blocks,
    count_section_headers,
    extract_phase_markers,
    detect_rfl_uplift_contexts,
    fingerprint_document,
    detect_missing_phase_markers,
    detect_uplifting_language,
    detect_code_drift,
    detect_terminology_drift,
    build_docs_drift_radar,
    validate_governance_annotations,
)


class TestNormalization:
    """Test text normalization functions."""
    
    def test_normalize_strips_trailing_whitespace(self):
        """Test that normalize_text strips trailing whitespace."""
        text = "line1   \nline2\t\nline3  "
        normalized = normalize_text(text)
        assert normalized == "line1\nline2\nline3"
    
    def test_normalize_collapses_blank_lines(self):
        """Test that normalize_text collapses multiple blank lines."""
        text = "line1\n\n\nline2\n\n\n\nline3"
        normalized = normalize_text(text)
        # Should have at most one blank line between content
        assert normalized.count('\n\n\n') == 0
    
    def test_normalize_preserves_indentation(self):
        """Test that normalize_text preserves leading whitespace."""
        text = "    indented\n  less indented\nno indent"
        normalized = normalize_text(text)
        assert "    indented" in normalized
        assert "  less indented" in normalized


class TestCodeBlockCounting:
    """Test code block counting."""
    
    def test_count_single_code_block(self):
        """Test counting a single code block."""
        text = "Some text\n```python\ncode\n```\nMore text"
        assert count_code_blocks(text) == 1
    
    def test_count_multiple_code_blocks(self):
        """Test counting multiple code blocks."""
        text = """
# Header
```python
code1
```

More text

```bash
code2
```
"""
        assert count_code_blocks(text) == 2
    
    def test_count_no_code_blocks(self):
        """Test counting when no code blocks exist."""
        text = "Just plain text\nNo code here"
        assert count_code_blocks(text) == 0


class TestSectionHeaderCounting:
    """Test section header counting."""
    
    def test_count_headers(self):
        """Test counting markdown headers."""
        text = """
# Header 1
## Header 2
### Header 3
Some text
## Another H2
"""
        assert count_section_headers(text) == 4
    
    def test_count_no_headers(self):
        """Test when no headers exist."""
        text = "Plain text\nNo headers"
        assert count_section_headers(text) == 0


class TestPhaseMarkerExtraction:
    """Test Phase marker extraction."""
    
    def test_extract_phase_ii_marker(self):
        """Test extracting PHASE II markers without matching PHASE I."""
        text = "This is PHASE II content for testing"
        markers = extract_phase_markers(text)
        # With word boundaries, "PHASE II" should not match "PHASE I"
        assert len(markers) == 1
        assert markers[0]['marker'] == 'PHASE II'
        assert 'PHASE II content' in markers[0]['context']
    
    def test_extract_multiple_phase_markers(self):
        """Test extracting multiple Phase markers."""
        text = """
PHASE I was completed.
Now we move to PHASE II.
Phase III is planned.
"""
        markers = extract_phase_markers(text)
        assert len(markers) >= 3
        marker_types = [m['marker'] for m in markers]
        assert 'PHASE I' in marker_types
        assert 'PHASE II' in marker_types
    
    def test_extract_no_markers(self):
        """Test when no Phase markers exist."""
        text = "Plain documentation with no phase markers"
        markers = extract_phase_markers(text)
        assert len(markers) == 0


class TestRFLContextDetection:
    """Test RFL/uplift context detection."""
    
    def test_detect_rfl_mention(self):
        """Test detecting RFL mentions."""
        text = "The RFL system provides reflexive learning"
        contexts = detect_rfl_uplift_contexts(text)
        assert len(contexts) >= 1
        assert any(ctx['term'] == 'RFL' for ctx in contexts)
    
    def test_detect_uplift_mention(self):
        """Test detecting uplift mentions."""
        text = "Performance uplift was observed in experiments"
        contexts = detect_rfl_uplift_contexts(text)
        assert len(contexts) >= 1
        assert any(ctx['term'] == 'uplift' for ctx in contexts)
    
    def test_detect_multiple_contexts(self):
        """Test detecting multiple RFL/uplift contexts."""
        text = "RFL provides policy guidance for uplift"
        contexts = detect_rfl_uplift_contexts(text)
        assert len(contexts) >= 3  # RFL, policy guidance, uplift


class TestDocumentFingerprinting:
    """Test document fingerprinting."""
    
    def test_fingerprint_valid_document(self):
        """Test fingerprinting a valid document."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Test Document

## PHASE II Content

```python
code_example()
```

This discusses RFL and uplift.
""")
            f.flush()
            path = Path(f.name)
        
        try:
            fp = fingerprint_document(path)
            
            assert 'error' not in fp
            assert fp['path'] == str(path)
            assert fp['code_blocks'] == 1
            assert fp['section_headers'] == 2
            assert len(fp['phase_markers']) >= 1
            assert len(fp['rfl_uplift_contexts']) >= 2
            assert 'content_hash' in fp
        finally:
            path.unlink()
    
    def test_fingerprint_nonexistent_file(self):
        """Test fingerprinting a non-existent file."""
        path = Path('/tmp/nonexistent_file_xyz.md')
        fp = fingerprint_document(path)
        assert 'error' in fp
        assert 'File not found' in fp['error']


class TestMissingPhaseMarkerDetection:
    """Test missing Phase marker detection."""
    
    def test_detect_missing_phase_ii_marker(self):
        """Test detecting RFL content without Phase II marker."""
        fingerprints = [
            {
                'path': 'test.md',
                'rfl_uplift_contexts': [
                    {'term': 'RFL', 'line': 'RFL provides guidance'},
                    {'term': 'uplift', 'line': 'uplift experiments'}
                ],
                'phase_markers': []  # No Phase II marker!
            }
        ]
        
        issues = detect_missing_phase_markers(fingerprints)
        assert len(issues) == 1
        assert issues[0]['type'] == 'missing_phase_ii_marker'
    
    def test_no_issue_when_marker_present(self):
        """Test no issue when Phase II marker is present."""
        fingerprints = [
            {
                'path': 'test.md',
                'rfl_uplift_contexts': [
                    {'term': 'RFL', 'line': 'RFL provides guidance'}
                ],
                'phase_markers': [
                    {'marker': 'PHASE II', 'context': 'PHASE II design'}
                ]
            }
        ]
        
        issues = detect_missing_phase_markers(fingerprints)
        assert len(issues) == 0


class TestUpliftingLanguageDetection:
    """Test uplifting language detection."""
    
    def test_detect_unguarded_uplift_claim(self):
        """Test detecting uplift claim without disclaimer."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("This system shows demonstrated uplift in production.")
            f.flush()
            path = Path(f.name)
        
        try:
            fp = fingerprint_document(path)
            issues = detect_uplifting_language([fp])
            
            assert len(issues) >= 1
            assert issues[0]['type'] == 'unguarded_uplift_claim'
            assert 'demonstrated uplift' in issues[0]['phrase']
        finally:
            path.unlink()
    
    def test_no_issue_with_disclaimer(self):
        """Test no issue when uplift claim has disclaimer."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""
> **STATUS: PHASE II — NOT YET RUN. NO UPLIFT CLAIMS MAY BE MADE.**

When run, this may show demonstrated uplift.
""")
            f.flush()
            path = Path(f.name)
        
        try:
            fp = fingerprint_document(path)
            issues = detect_uplifting_language([fp])
            
            # Should find the phrase but also the disclaimer
            # Our logic checks within 500 chars, so should not report issue
            assert len(issues) == 0
        finally:
            path.unlink()


class TestCodeDriftDetection:
    """Test code drift detection."""
    
    def test_detect_code_block_increase(self):
        """Test detecting increase in code blocks."""
        fingerprints = [
            {'path': 'v1.md', 'code_blocks': 2},
            {'path': 'v2.md', 'code_blocks': 5}
        ]
        
        issues = detect_code_drift(fingerprints)
        assert len(issues) == 1
        assert issues[0]['type'] == 'code_block_count_change'
        assert issues[0]['delta'] == 3
    
    def test_no_drift_when_unchanged(self):
        """Test no drift when code blocks unchanged."""
        fingerprints = [
            {'path': 'v1.md', 'code_blocks': 3},
            {'path': 'v2.md', 'code_blocks': 3}
        ]
        
        issues = detect_code_drift(fingerprints)
        assert len(issues) == 0


class TestTerminologyDriftDetection:
    """Test terminology drift detection."""
    
    def test_detect_terminology_expansion(self):
        """Test detecting significant increase in RFL terminology."""
        fingerprints = [
            {'path': 'v1.md', 'rfl_uplift_contexts': [{'term': 'RFL'}, {'term': 'uplift'}]},
            {'path': 'v2.md', 'rfl_uplift_contexts': [
                {'term': 'RFL'}, {'term': 'uplift'}, {'term': 'RFL'},
                {'term': 'policy guidance'}, {'term': 'uplift'}, {'term': 'RFL'}
            ]}
        ]
        
        issues = detect_terminology_drift(fingerprints)
        assert len(issues) == 1
        assert issues[0]['type'] == 'terminology_expansion'
        assert issues[0]['delta'] > 0


class TestDriftRadar:
    """Test comprehensive drift radar."""
    
    def test_build_drift_radar(self):
        """Test building complete drift radar."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f1:
            f1.write("# Doc v1\n```python\ncode\n```\nRFL content")
            f1.flush()
            path1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f2:
            f2.write("# Doc v2\n```python\ncode1\n```\n```python\ncode2\n```\nRFL and uplift content here")
            f2.flush()
            path2 = Path(f2.name)
        
        try:
            fp1 = fingerprint_document(path1)
            fp2 = fingerprint_document(path2)
            
            radar = build_docs_drift_radar([fp1, fp2])
            
            assert radar['format_version'] == '1.0'
            assert radar['report_type'] == 'docs_drift_radar'
            assert radar['documents_analyzed'] == 2
            assert 'issues' in radar
            assert 'missing_phase_markers' in radar['issues']
            assert 'code_drift' in radar['issues']
            assert 'terminology_drift' in radar['issues']
        finally:
            path1.unlink()
            path2.unlink()


class TestGovernanceValidation:
    """Test governance annotation validation."""
    
    def test_missing_disclaimer_on_phase_ii(self):
        """Test detecting Phase II content without disclaimer."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            # Use explicit "PHASE II" marker in a way that won't accidentally match disclaimers
            f.write("# System Design for PHASE II\n\nThis describes PHASE II architecture and features.")
            f.flush()
            path = Path(f.name)
        
        try:
            fp = fingerprint_document(path)
            validation = validate_governance_annotations([fp])
            
            assert validation['status'] == 'FAIL'
            assert validation['total_issues'] >= 1
            assert any(
                issue['type'] == 'missing_disclaimer'
                for issue in validation['issues']
            )
        finally:
            path.unlink()
    
    def test_pass_with_proper_disclaimer(self):
        """Test passing validation with proper disclaimer."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# PHASE II Design

> **STATUS: PHASE II — NOT YET RUN. NO UPLIFT CLAIMS MAY BE MADE.**

This is Phase II content with proper disclaimer.
""")
            f.flush()
            path = Path(f.name)
        
        try:
            fp = fingerprint_document(path)
            validation = validate_governance_annotations([fp])
            
            assert validation['status'] == 'PASS'
            assert validation['total_issues'] == 0
        finally:
            path.unlink()
    
    def test_detect_premature_readiness_claim(self):
        """Test detecting premature readiness claims."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# PHASE II System

> **STATUS: PHASE II — NOT YET RUN.**

The system is uplift ready for production deployment.
""")
            f.flush()
            path = Path(f.name)
        
        try:
            fp = fingerprint_document(path)
            validation = validate_governance_annotations([fp])
            
            assert validation['total_issues'] >= 1
            assert any(
                issue['type'] == 'premature_readiness_claim'
                for issue in validation['issues']
            )
        finally:
            path.unlink()


class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_docs_fingerprint(self):
        """Test --docs-fingerprint CLI argument."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test\nSome content")
            f.flush()
            path = Path(f.name)
        
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    'docs/docs_fingerprint.py',
                    '--docs-fingerprint', str(path)
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            assert result.returncode == 0
            # Check that stdout contains expected output (may have debug info before JSON)
            assert 'content_hash' in result.stdout
            assert str(path) in result.stdout
        finally:
            path.unlink()
    
    def test_cli_docs_drift_history(self):
        """Test --docs-drift-history CLI argument."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f1:
            f1.write("# V1\n```code```")
            f1.flush()
            path1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f2:
            f2.write("# V2\n```code1```\n```code2```")
            f2.flush()
            path2 = Path(f2.name)
        
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    'docs/docs_fingerprint.py',
                    '--docs-drift-history', str(path1), str(path2)
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            # May return 1 if drift detected
            assert result.returncode in [0, 1]
            assert 'Drift Radar Results' in result.stdout
        finally:
            path1.unlink()
            path2.unlink()
    
    def test_cli_docs_validate_governance(self):
        """Test --docs-validate-governance CLI argument."""
        # Create temp docs directory
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / 'docs'
            docs_dir.mkdir()
            
            (docs_dir / 'test.md').write_text("# Test\nRegular content")
            
            result = subprocess.run(
                [
                    sys.executable,
                    'docs/docs_fingerprint.py',
                    '--docs-validate-governance',
                    '--docs-dir', str(docs_dir)
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            assert result.returncode == 0  # Should pass for regular content
            assert 'Governance Validation Results' in result.stdout


@pytest.mark.parametrize("text,expected", [
    ("line1\nline2", "line1\nline2"),
    ("  spaced  \n  lines  ", "  spaced\n  lines"),
    ("a\n\n\nb", "a\n\nb"),  # Collapse blank lines
])
def test_normalize_text_parametrized(text, expected):
    """Parametrized test for text normalization."""
    assert normalize_text(text) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
