"""
Tests for emit_budget_confounding_truth_table CLI script.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from derivation.budget_cal_exp_integration import build_budget_confounding_truth_table


def test_emit_truth_table_default_threshold():
    """Test CLI emits truth table with default threshold."""
    result = subprocess.run(
        [sys.executable, "scripts/emit_budget_confounding_truth_table.py", "--json"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    
    assert result.returncode == 0
    
    # Parse JSON output
    table = json.loads(result.stdout)
    
    # Verify structure
    assert table["schema_version"] == "1.0.0"
    assert table["mode"] == "SHADOW"
    assert table["confound_stability_threshold"] == 0.95
    assert len(table["truth_table"]) == 6


def test_emit_truth_table_custom_threshold():
    """Test CLI emits truth table with custom threshold."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/emit_budget_confounding_truth_table.py",
            "--json",
            "--threshold",
            "0.90",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    
    assert result.returncode == 0
    
    # Parse JSON output
    table = json.loads(result.stdout)
    
    # Verify custom threshold
    assert table["confound_stability_threshold"] == 0.90
    assert len(table["truth_table"]) == 6


def test_emit_truth_table_deterministic():
    """Test CLI output is deterministic."""
    result1 = subprocess.run(
        [sys.executable, "scripts/emit_budget_confounding_truth_table.py", "--json"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    
    result2 = subprocess.run(
        [sys.executable, "scripts/emit_budget_confounding_truth_table.py", "--json"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    
    # Outputs should be byte-identical
    assert result1.stdout == result2.stdout
    
    # Should match direct function call
    table_func = build_budget_confounding_truth_table()
    table_cli = json.loads(result1.stdout)
    
    assert table_func == table_cli

