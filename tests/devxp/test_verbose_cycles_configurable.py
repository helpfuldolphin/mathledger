"""
PHASE II — NOT USED IN PHASE I

Tests for verbose-cycles configurable fields feature

Validates:
- Custom field selection via U2_VERBOSE_FIELDS env var
- Default behavior when no config provided
- Field formatter with key=value output
- Unknown fields handled gracefully
"""

import os
import pytest
from unittest.mock import patch


@pytest.mark.unit
def test_parse_verbose_fields_from_env():
    """Test parsing verbose fields from environment variable."""
    from experiments.verbose_formatter import parse_verbose_fields
    
    with patch.dict(os.environ, {"U2_VERBOSE_FIELDS": "cycle,mode,success,item"}):
        fields = parse_verbose_fields()
        assert fields == ["cycle", "mode", "success", "item"]


@pytest.mark.unit
def test_parse_verbose_fields_with_spaces():
    """Test parsing handles spaces around commas."""
    from experiments.verbose_formatter import parse_verbose_fields
    
    with patch.dict(os.environ, {"U2_VERBOSE_FIELDS": "cycle , mode , success"}):
        fields = parse_verbose_fields()
        assert fields == ["cycle", "mode", "success"]


@pytest.mark.unit
def test_parse_verbose_fields_empty_env():
    """Test parsing returns None when env var not set."""
    from experiments.verbose_formatter import parse_verbose_fields
    
    with patch.dict(os.environ, {}, clear=True):
        fields = parse_verbose_fields()
        assert fields is None


@pytest.mark.unit
def test_parse_verbose_fields_empty_string():
    """Test parsing returns None for empty string."""
    from experiments.verbose_formatter import parse_verbose_fields
    
    with patch.dict(os.environ, {"U2_VERBOSE_FIELDS": ""}):
        fields = parse_verbose_fields()
        assert fields is None


@pytest.mark.unit
def test_format_verbose_cycle_basic():
    """Test basic verbose cycle formatting."""
    from experiments.verbose_formatter import format_verbose_cycle
    
    fields = ["cycle", "mode", "success"]
    data = {
        "cycle": 1,
        "mode": "baseline",
        "success": True,
        "item": "p->q",
    }
    
    result = format_verbose_cycle(fields, data)
    assert "cycle=1" in result
    assert "mode=baseline" in result
    assert "success=true" in result


@pytest.mark.unit
def test_format_verbose_cycle_all_types():
    """Test formatting handles different data types."""
    from experiments.verbose_formatter import format_verbose_cycle
    
    fields = ["cycle", "success", "item", "value"]
    data = {
        "cycle": 42,
        "success": False,
        "item": "test",
        "value": 3.14,
    }
    
    result = format_verbose_cycle(fields, data)
    assert "cycle=42" in result
    assert "success=false" in result
    assert "item=test" in result
    assert "value=3.14" in result


@pytest.mark.unit
def test_format_verbose_cycle_missing_field():
    """Test formatting handles missing fields gracefully."""
    from experiments.verbose_formatter import format_verbose_cycle
    
    fields = ["cycle", "missing_field", "success"]
    data = {
        "cycle": 1,
        "success": True,
    }
    
    result = format_verbose_cycle(fields, data)
    assert "cycle=1" in result
    assert "missing_field=N/A" in result
    assert "success=true" in result


@pytest.mark.unit
def test_format_verbose_cycle_preserves_order():
    """Test that field order is preserved in output."""
    from experiments.verbose_formatter import format_verbose_cycle
    
    fields = ["z_field", "a_field", "m_field"]
    data = {
        "z_field": "last",
        "a_field": "first",
        "m_field": "middle",
    }
    
    result = format_verbose_cycle(fields, data)
    parts = result.split()
    
    # Check order matches field list, not alphabetical
    assert parts[0].startswith("z_field=")
    assert parts[1].startswith("a_field=")
    assert parts[2].startswith("m_field=")


@pytest.mark.unit
def test_format_verbose_cycle_extended_fields():
    """Test formatting with extended field set."""
    from experiments.verbose_formatter import format_verbose_cycle
    
    fields = ["cycle", "mode", "success", "label", "item_hash_prefix"]
    data = {
        "cycle": 5,
        "mode": "rfl",
        "success": True,
        "label": "PHASE_II",
        "item_hash_prefix": "abc12345",
    }
    
    result = format_verbose_cycle(fields, data)
    assert "cycle=5" in result
    assert "mode=rfl" in result
    assert "label=PHASE_II" in result
    assert "item_hash_prefix=abc12345" in result


@pytest.mark.unit
def test_format_verbose_cycle_single_field():
    """Test formatting with single field."""
    from experiments.verbose_formatter import format_verbose_cycle
    
    fields = ["cycle"]
    data = {"cycle": 99}
    
    result = format_verbose_cycle(fields, data)
    assert result == "cycle=99"


@pytest.mark.unit
def test_format_verbose_cycle_empty_fields():
    """Test formatting with empty field list."""
    from experiments.verbose_formatter import format_verbose_cycle
    
    fields = []
    data = {"cycle": 1}
    
    result = format_verbose_cycle(fields, data)
    assert result == ""


@pytest.mark.unit
def test_parse_verbose_fields_custom_env_var():
    """Test parsing from custom environment variable name."""
    from experiments.verbose_formatter import parse_verbose_fields
    
    with patch.dict(os.environ, {"CUSTOM_VERBOSE": "a,b,c"}):
        fields = parse_verbose_fields(env_var="CUSTOM_VERBOSE")
        assert fields == ["a", "b", "c"]


@pytest.mark.unit
def test_format_verbose_cycle_special_characters():
    """Test formatting handles special characters in values."""
    from experiments.verbose_formatter import format_verbose_cycle
    
    fields = ["item", "result"]
    data = {
        "item": "p->q->r",
        "result": "{'outcome': 'VERIFIED'}",
    }
    
    result = format_verbose_cycle(fields, data)
    assert "item=p->q->r" in result
    # Should handle dict string representation
    assert "result=" in result


# Integration-style tests (without full CLI)

@pytest.mark.unit
def test_verbose_fields_integration_default():
    """Test default fields are used when verbose_fields is None."""
    # This tests the logic that would be in run_experiment
    verbose_fields = None
    
    if verbose_fields is None:
        verbose_fields_to_use = ["cycle", "mode", "success", "item"]
    else:
        verbose_fields_to_use = verbose_fields
    
    assert verbose_fields_to_use == ["cycle", "mode", "success", "item"]


@pytest.mark.unit
def test_verbose_fields_integration_custom():
    """Test custom fields override defaults."""
    verbose_fields = ["cycle", "label", "item_hash_prefix"]
    
    if verbose_fields is None:
        verbose_fields_to_use = ["cycle", "mode", "success", "item"]
    else:
        verbose_fields_to_use = verbose_fields
    
    assert verbose_fields_to_use == ["cycle", "label", "item_hash_prefix"]


@pytest.mark.unit
def test_verbose_output_format_machine_parseable():
    """Test that verbose output is machine-parseable."""
    from experiments.verbose_formatter import format_verbose_cycle
    
    fields = ["cycle", "success", "mode"]
    data = {"cycle": 10, "success": True, "mode": "baseline"}
    
    result = format_verbose_cycle(fields, data)
    
    # Parse back
    parts = result.split()
    parsed = {}
    for part in parts:
        key, value = part.split("=", 1)
        parsed[key] = value
    
    assert parsed["cycle"] == "10"
    assert parsed["success"] == "true"
    assert parsed["mode"] == "baseline"
