#!/usr/bin/env python3
"""
Tests for docs/generate_evidence_pack_toc.py - Evidence Pack TOC generator.

Tests:
- Config loading (YAML/JSON)
- Entry validation
- JSON TOC generation
- Markdown TOC generation
- Metadata enrichment
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from docs.generate_evidence_pack_toc import EvidencePackTOCGenerator


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure for testing."""
    # Create some documentation files
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    (docs_dir / "README.md").write_text("# Main README\n")
    (docs_dir / "ARCHITECTURE.md").write_text("# Architecture\n")
    (docs_dir / "PHASE2_PLAN.md").write_text("# Phase II Plan\n")
    
    return tmp_path


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'metadata': {
            'version': '1.0',
            'description': 'Test Evidence Pack',
            'date': '2025-12'
        },
        'documents': [
            {
                'path': 'docs/README.md',
                'description': 'Main README file',
                'phase': 'General',
                'category': 'readme',
                'tags': ['readme', 'introduction']
            },
            {
                'path': 'docs/ARCHITECTURE.md',
                'description': 'Architecture documentation',
                'phase': 'Phase I',
                'category': 'architecture',
                'tags': ['architecture', 'design']
            },
            {
                'path': 'docs/PHASE2_PLAN.md',
                'description': 'Phase II planning document',
                'phase': 'Phase II',
                'category': 'planning',
                'tags': ['phase-ii', 'planning']
            }
        ]
    }


def test_load_yaml_config(temp_project, sample_config):
    """Test loading YAML configuration."""
    config_path = temp_project / "config.yaml"
    config_path.write_text(yaml.dump(sample_config), encoding='utf-8')
    
    generator = EvidencePackTOCGenerator(temp_project)
    loaded = generator.load_config(config_path)
    
    assert loaded['metadata']['version'] == '1.0'
    assert len(loaded['documents']) == 3


def test_load_json_config(temp_project, sample_config):
    """Test loading JSON configuration."""
    config_path = temp_project / "config.json"
    config_path.write_text(json.dumps(sample_config), encoding='utf-8')
    
    generator = EvidencePackTOCGenerator(temp_project)
    loaded = generator.load_config(config_path)
    
    assert loaded['metadata']['version'] == '1.0'
    assert len(loaded['documents']) == 3


def test_validate_entry_success(temp_project):
    """Test validation of valid entry."""
    generator = EvidencePackTOCGenerator(temp_project)
    
    entry = {
        'path': 'docs/README.md',
        'description': 'Main README'
    }
    
    assert generator.validate_entry(entry) is True


def test_validate_entry_missing_field(temp_project):
    """Test validation fails for missing required field."""
    generator = EvidencePackTOCGenerator(temp_project)
    
    entry = {
        'path': 'docs/README.md'
        # Missing 'description'
    }
    
    assert generator.validate_entry(entry) is False


def test_validate_entry_missing_file(temp_project):
    """Test validation fails for non-existent file."""
    generator = EvidencePackTOCGenerator(temp_project)
    
    entry = {
        'path': 'docs/NONEXISTENT.md',
        'description': 'This file does not exist'
    }
    
    assert generator.validate_entry(entry) is False


def test_add_metadata(temp_project):
    """Test adding metadata to entry."""
    generator = EvidencePackTOCGenerator(temp_project)
    
    entry = {
        'path': 'docs/README.md',
        'description': 'Main README'
    }
    
    enriched = generator.add_metadata(entry)
    
    assert 'file_size' in enriched
    assert 'modified_time' in enriched
    assert enriched['file_size'] > 0


def test_process_config(temp_project, sample_config):
    """Test processing configuration."""
    generator = EvidencePackTOCGenerator(temp_project)
    entries = generator.process_config(sample_config)
    
    # All three files exist, so all should be processed
    assert len(entries) == 3
    
    # Check metadata was added
    for entry in entries:
        assert 'file_size' in entry
        assert 'modified_time' in entry


def test_process_config_skips_invalid(temp_project):
    """Test that invalid entries are skipped."""
    config = {
        'documents': [
            {
                'path': 'docs/README.md',
                'description': 'Valid entry'
            },
            {
                'path': 'docs/MISSING.md',
                'description': 'Invalid entry - file missing'
            },
            {
                'path': 'docs/ARCHITECTURE.md'
                # Invalid - missing description
            }
        ]
    }
    
    generator = EvidencePackTOCGenerator(temp_project)
    entries = generator.process_config(config)
    
    # Only the valid entry should be included
    assert len(entries) == 1
    assert entries[0]['path'] == 'docs/README.md'


def test_generate_json(temp_project, sample_config):
    """Test JSON TOC generation."""
    generator = EvidencePackTOCGenerator(temp_project)
    entries = generator.process_config(sample_config)
    
    toc = generator.generate_json(entries, sample_config['metadata'])
    
    assert toc['format_version'] == '1.0'
    assert 'generated_at' in toc
    assert toc['metadata']['version'] == '1.0'
    assert len(toc['documents']) == 3


def test_generate_markdown(temp_project, sample_config):
    """Test Markdown TOC generation."""
    generator = EvidencePackTOCGenerator(temp_project)
    entries = generator.process_config(sample_config)
    
    md = generator.generate_markdown(entries, sample_config['metadata'])
    
    # Check headers
    assert '# Evidence Pack v1 - Table of Contents' in md
    assert '**Version:** 1.0' in md
    assert '**Description:** Test Evidence Pack' in md
    
    # Check document entries
    assert 'docs/README.md' in md
    assert 'Main README file' in md
    assert 'docs/ARCHITECTURE.md' in md
    
    # Check phases
    assert '## General' in md
    assert '## Phase I' in md
    assert '## Phase II' in md


def test_generate_markdown_groups_by_phase(temp_project, sample_config):
    """Test that markdown groups documents by phase."""
    generator = EvidencePackTOCGenerator(temp_project)
    entries = generator.process_config(sample_config)
    
    md = generator.generate_markdown(entries, sample_config['metadata'])
    
    # Find phase sections
    general_idx = md.index('## General')
    phase1_idx = md.index('## Phase I')
    phase2_idx = md.index('## Phase II')
    
    # Check ordering (alphabetical)
    assert general_idx < phase1_idx < phase2_idx
    
    # Check documents appear under correct phase
    readme_idx = md.index('docs/README.md')
    arch_idx = md.index('docs/ARCHITECTURE.md')
    plan_idx = md.index('docs/PHASE2_PLAN.md')
    
    assert general_idx < readme_idx < phase1_idx
    assert phase1_idx < arch_idx < phase2_idx
    assert phase2_idx < plan_idx


def test_generate_full_workflow(temp_project, sample_config):
    """Test full generation workflow."""
    # Create config file
    config_path = temp_project / "config.yaml"
    config_path.write_text(yaml.dump(sample_config), encoding='utf-8')
    
    # Create output directory
    output_dir = temp_project / "output"
    
    # Generate TOC
    generator = EvidencePackTOCGenerator(temp_project)
    success = generator.generate(config_path, output_dir)
    
    assert success is True
    
    # Check JSON output
    json_output = output_dir / "evidence_pack_v1_toc.json"
    assert json_output.exists()
    
    json_data = json.loads(json_output.read_text())
    assert json_data['format_version'] == '1.0'
    assert len(json_data['documents']) == 3
    
    # Check Markdown output
    md_output = output_dir / "evidence_pack_v1_toc.md"
    assert md_output.exists()
    
    md_content = md_output.read_text()
    assert 'Evidence Pack v1' in md_content
    assert 'docs/README.md' in md_content


def test_generate_empty_config_fails(temp_project):
    """Test that empty config fails gracefully."""
    config = {'documents': []}
    
    config_path = temp_project / "empty_config.yaml"
    config_path.write_text(yaml.dump(config), encoding='utf-8')
    
    output_dir = temp_project / "output"
    
    generator = EvidencePackTOCGenerator(temp_project)
    success = generator.generate(config_path, output_dir)
    
    assert success is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
