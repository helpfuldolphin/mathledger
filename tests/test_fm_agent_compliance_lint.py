"""
PHASE IV — NOT RUN IN PHASE I

Tests for Agent Compliance Linter

This module contains tests covering:
  - Label extraction from agent configs
  - Compliance checking logic
  - Status determination (OK, ATTENTION, BLOCK)

ABSOLUTE SAFEGUARDS:
    - Tests are DESCRIPTIVE, not NORMATIVE
    - No modifications to agent configs
    - No inference or claims regarding uplift
"""

import json
import pytest
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fm_agent_compliance_lint import (
    extract_labels_from_text,
    extract_labels_from_agent_config,
    get_fm_labels_from_contract_index,
    lint_agent_compliance,
)


class TestLabelExtraction:
    """Tests for label extraction from text and configs."""
    
    def test_extract_labels_from_text_basic(self):
        """extract_labels_from_text extracts basic label patterns."""
        content = "This uses def:slice and inv:monotonicity."
        labels = extract_labels_from_text(content)
        
        assert "def:slice" in labels
        assert "inv:monotonicity" in labels
    
    def test_extract_labels_from_text_latex_refs(self):
        """extract_labels_from_text extracts LaTeX refs."""
        content = "See \\ref{def:slice} and \\eqref{eq:main}."
        labels = extract_labels_from_text(content)
        
        assert "def:slice" in labels
        assert "eq:main" in labels
    
    def test_extract_labels_from_text_quoted(self):
        """extract_labels_from_text extracts quoted labels."""
        content = 'The "def:policy" is important.'
        labels = extract_labels_from_text(content)
        
        assert "def:policy" in labels
    
    def test_extract_labels_from_agent_config_json(self):
        """extract_labels_from_agent_config extracts from JSON."""
        config_data = {
            "prompt": "Use def:slice in your analysis.",
            "rules": ["Check inv:monotonicity", "Verify def:policy"],
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            labels = extract_labels_from_agent_config(config_path)
            
            assert "def:slice" in labels
            assert "inv:monotonicity" in labels
            assert "def:policy" in labels
        finally:
            config_path.unlink()
    
    def test_extract_labels_from_agent_config_text(self):
        """extract_labels_from_agent_config extracts from text files."""
        content = "Agent should use def:slice and inv:monotonicity."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            config_path = Path(f.name)
        
        try:
            labels = extract_labels_from_agent_config(config_path)
            
            assert "def:slice" in labels
            assert "inv:monotonicity" in labels
        finally:
            config_path.unlink()


class TestComplianceLinting:
    """Tests for agent compliance linting."""
    
    @pytest.fixture
    def sample_label_index(self) -> Dict[str, Any]:
        """Sample label contract index."""
        return {
            "schema_version": "1.0.0",
            "label_index": {
                "def:slice": {"in_fm": True, "in_taxonomy": False, "in_curriculum": False},
                "def:policy": {"in_fm": True, "in_taxonomy": False, "in_curriculum": False},
                "inv:monotonicity": {"in_fm": True, "in_taxonomy": False, "in_curriculum": False},
            },
            "labels_missing_in_taxonomy": [],
            "labels_missing_in_curriculum": [],
            "labels_missing_in_fm": [],
            "contract_status": "ALIGNED",
        }
    
    def test_lint_agent_compliance_ok(self, sample_label_index):
        """lint_agent_compliance returns OK when all FM labels are used."""
        # Create a label index where agent uses all FM labels
        complete_label_index = {
            "schema_version": "1.0.0",
            "label_index": {
                "def:slice": {"in_fm": True, "in_taxonomy": False, "in_curriculum": False},
                "def:policy": {"in_fm": True, "in_taxonomy": False, "in_curriculum": False},
            },
            "labels_missing_in_taxonomy": [],
            "labels_missing_in_curriculum": [],
            "labels_missing_in_fm": [],
            "contract_status": "ALIGNED",
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Agent uses def:slice and def:policy.")
            agent_path = Path(f.name)
        
        try:
            compliance = lint_agent_compliance(complete_label_index, [agent_path])
            
            assert compliance["status"] == "OK"
            assert len(compliance["missing_labels"]) == 0
            assert len(compliance["superfluous_labels"]) == 0
            assert len(compliance["agents_checked"]) == 1
        finally:
            agent_path.unlink()
    
    def test_lint_agent_compliance_block_missing_labels(self, sample_label_index):
        """lint_agent_compliance returns BLOCK for missing FM labels."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Agent uses def:missing_label.")
            agent_path = Path(f.name)
        
        try:
            compliance = lint_agent_compliance(sample_label_index, [agent_path])
            
            assert compliance["status"] == "BLOCK"
            assert "def:missing_label" in compliance["missing_labels"]
            assert len(compliance["agents_checked"]) == 1
        finally:
            agent_path.unlink()
    
    def test_lint_agent_compliance_attention_superfluous(self, sample_label_index):
        """lint_agent_compliance returns ATTENTION for superfluous labels."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Agent uses def:slice.")
            agent_path = Path(f.name)
        
        try:
            compliance = lint_agent_compliance(sample_label_index, [agent_path])
            
            # Should be ATTENTION because FM has labels not used by agent
            assert compliance["status"] == "ATTENTION"
            assert len(compliance["superfluous_labels"]) > 0
        finally:
            agent_path.unlink()
    
    def test_lint_agent_compliance_schema(self, sample_label_index):
        """lint_agent_compliance returns correct schema."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Agent uses def:slice.")
            agent_path = Path(f.name)
        
        try:
            compliance = lint_agent_compliance(sample_label_index, [agent_path])
            
            assert "schema_version" in compliance
            assert "agents_checked" in compliance
            assert "missing_labels" in compliance
            assert "superfluous_labels" in compliance
            assert "status" in compliance
            assert "neutral_notes" in compliance
        finally:
            agent_path.unlink()
    
    def test_lint_agent_compliance_multiple_agents(self, sample_label_index):
        """lint_agent_compliance handles multiple agent configs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("Agent1 uses def:slice.")
            agent1_path = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("Agent2 uses def:policy.")
            agent2_path = Path(f2.name)
        
        try:
            compliance = lint_agent_compliance(
                sample_label_index, [agent1_path, agent2_path]
            )
            
            assert len(compliance["agents_checked"]) == 2
            assert compliance["agents_checked"][0]["agent"] == str(agent1_path)
            assert compliance["agents_checked"][1]["agent"] == str(agent2_path)
        finally:
            agent1_path.unlink()
            agent2_path.unlink()
    
    def test_lint_agent_compliance_deterministic(self, sample_label_index):
        """lint_agent_compliance is deterministic."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Agent uses def:slice.")
            agent_path = Path(f.name)
        
        try:
            compliance1 = lint_agent_compliance(sample_label_index, [agent_path])
            compliance2 = lint_agent_compliance(sample_label_index, [agent_path])
            
            assert compliance1 == compliance2
        finally:
            agent_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

