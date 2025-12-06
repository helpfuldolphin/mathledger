"""
PHASE II — NOT USED IN PHASE I

Tests for curriculum_loader_v2.py

Validates:
- Deterministic ordering of curriculum items
- Error handling for missing/malformed files
- Support for YAML/JSON/JSONL formats
- Slice lookup and metadata extraction
"""

import json
import pytest
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.curriculum_loader_v2 import (
    CurriculumLoader,
    CurriculumItem,
    CurriculumLoaderError,
    CurriculumNotFoundError,
    CurriculumFormatError,
    load_curriculum_for_slice,
)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_curriculum_dict():
    """Sample curriculum config as a dictionary."""
    return {
        "version": "2.1.0",
        "slices": {
            "test_slice_a": {
                "description": "Test slice A",
                "formula_pool_entries": [
                    "p",
                    "q",
                    "p->q",
                    "q->p",
                ]
            },
            "test_slice_b": {
                "description": "Test slice B",
                "formula_pool_entries": [
                    {"formula": "p|q", "hash": "abc123"},
                    {"formula": "p&q", "hash": "def456"},
                ]
            }
        }
    }


@pytest.fixture
def sample_yaml_file(temp_dir, sample_curriculum_dict):
    """Create a sample YAML curriculum file."""
    yaml_file = temp_dir / "curriculum.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(sample_curriculum_dict, f)
    return yaml_file


@pytest.fixture
def sample_json_file(temp_dir, sample_curriculum_dict):
    """Create a sample JSON curriculum file."""
    json_file = temp_dir / "curriculum.json"
    with open(json_file, 'w') as f:
        json.dump(sample_curriculum_dict, f, indent=2)
    return json_file


def test_loader_initialization_success(sample_yaml_file):
    """Test successful loader initialization with valid file."""
    loader = CurriculumLoader(sample_yaml_file)
    assert loader.config_path == sample_yaml_file
    assert loader._config is not None


def test_loader_initialization_file_not_found(temp_dir):
    """Test loader raises error when file doesn't exist."""
    nonexistent = temp_dir / "nonexistent.yaml"
    with pytest.raises(CurriculumNotFoundError) as exc_info:
        CurriculumLoader(nonexistent)
    assert "not found" in str(exc_info.value).lower()


def test_loader_yaml_format(sample_yaml_file):
    """Test loading YAML format."""
    loader = CurriculumLoader(sample_yaml_file)
    items = loader.load_for_slice("test_slice_a")
    assert len(items) == 4
    assert items[0].formula == "p"
    assert items[2].formula == "p->q"


def test_loader_json_format(sample_json_file):
    """Test loading JSON format."""
    loader = CurriculumLoader(sample_json_file)
    items = loader.load_for_slice("test_slice_a")
    assert len(items) == 4
    assert items[0].formula == "p"


def test_deterministic_ordering(sample_yaml_file):
    """Test that item ordering is deterministic across multiple loads."""
    loader1 = CurriculumLoader(sample_yaml_file)
    items1 = loader1.load_for_slice("test_slice_a")
    
    loader2 = CurriculumLoader(sample_yaml_file)
    items2 = loader2.load_for_slice("test_slice_a")
    
    # Verify same length
    assert len(items1) == len(items2)
    
    # Verify same order
    for i, (item1, item2) in enumerate(zip(items1, items2)):
        assert item1.formula == item2.formula, f"Order mismatch at index {i}"


def test_load_with_metadata(sample_yaml_file):
    """Test loading items with metadata (dict format)."""
    loader = CurriculumLoader(sample_yaml_file)
    items = loader.load_for_slice("test_slice_b")
    
    assert len(items) == 2
    assert items[0].formula == "p|q"
    assert items[0].hash == "abc123"
    assert items[1].formula == "p&q"
    assert items[1].hash == "def456"


def test_slice_not_found(sample_yaml_file):
    """Test error when requesting non-existent slice."""
    loader = CurriculumLoader(sample_yaml_file)
    with pytest.raises(CurriculumNotFoundError) as exc_info:
        loader.load_for_slice("nonexistent_slice")
    
    error_msg = str(exc_info.value)
    assert "not found" in error_msg.lower()
    assert "test_slice_a" in error_msg or "test_slice_b" in error_msg


def test_malformed_yaml(temp_dir):
    """Test error handling for malformed YAML."""
    bad_yaml = temp_dir / "bad.yaml"
    with open(bad_yaml, 'w') as f:
        f.write("{\nthis is: [not valid yaml\n")
    
    with pytest.raises(CurriculumFormatError) as exc_info:
        CurriculumLoader(bad_yaml)
    assert "parse" in str(exc_info.value).lower()


def test_malformed_json(temp_dir):
    """Test error handling for malformed JSON."""
    bad_json = temp_dir / "bad.json"
    with open(bad_json, 'w') as f:
        f.write('{"slices": {invalid json}')
    
    with pytest.raises(CurriculumFormatError) as exc_info:
        CurriculumLoader(bad_json)
    assert "parse" in str(exc_info.value).lower()


def test_missing_slices_key(temp_dir):
    """Test error when config is missing 'slices' key."""
    bad_config = temp_dir / "no_slices.yaml"
    with open(bad_config, 'w') as f:
        yaml.dump({"version": "1.0"}, f)
    
    with pytest.raises(CurriculumFormatError) as exc_info:
        CurriculumLoader(bad_config)
    assert "slices" in str(exc_info.value).lower()


def test_empty_formula_pool(temp_dir):
    """Test error when slice has no formula entries."""
    config = {
        "slices": {
            "empty_slice": {
                "description": "No formulas",
                "formula_pool_entries": []
            }
        }
    }
    config_file = temp_dir / "empty.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    loader = CurriculumLoader(config_file)
    with pytest.raises(CurriculumFormatError) as exc_info:
        loader.load_for_slice("empty_slice")
    assert "no formula" in str(exc_info.value).lower()


def test_list_slices(sample_yaml_file):
    """Test listing available slices."""
    loader = CurriculumLoader(sample_yaml_file)
    slices = loader.list_slices()
    
    assert isinstance(slices, list)
    assert "test_slice_a" in slices
    assert "test_slice_b" in slices
    assert slices == sorted(slices)  # Should be sorted


def test_get_slice_config(sample_yaml_file):
    """Test getting full slice configuration."""
    loader = CurriculumLoader(sample_yaml_file)
    config = loader.get_slice_config("test_slice_a")
    
    assert isinstance(config, dict)
    assert config["description"] == "Test slice A"
    assert "formula_pool_entries" in config


def test_convenience_function(sample_yaml_file):
    """Test the convenience load_curriculum_for_slice function."""
    items = load_curriculum_for_slice(sample_yaml_file, "test_slice_a")
    
    assert len(items) == 4
    assert all(isinstance(item, CurriculumItem) for item in items)


def test_index_metadata_preserved(sample_yaml_file):
    """Test that index metadata is added and preserved."""
    loader = CurriculumLoader(sample_yaml_file)
    items = loader.load_for_slice("test_slice_a")
    
    for i, item in enumerate(items):
        assert item.metadata is not None
        assert item.metadata["index"] == i


def test_jsonl_format(temp_dir):
    """Test loading JSONL format (single object)."""
    config = {
        "slices": {
            "test_slice": {
                "formula_pool_entries": ["p", "q", "r"]
            }
        }
    }
    jsonl_file = temp_dir / "curriculum.jsonl"
    with open(jsonl_file, 'w') as f:
        f.write(json.dumps(config) + '\n')
    
    loader = CurriculumLoader(jsonl_file)
    items = loader.load_for_slice("test_slice")
    assert len(items) == 3


def test_unsupported_format(temp_dir):
    """Test error for unsupported file format."""
    bad_file = temp_dir / "curriculum.txt"
    with open(bad_file, 'w') as f:
        f.write("some text")
    
    with pytest.raises(CurriculumLoaderError) as exc_info:
        CurriculumLoader(bad_file)
    assert "unsupported" in str(exc_info.value).lower()


def test_real_curriculum_file():
    """Test loading the actual Phase II curriculum config."""
    config_path = Path("/home/runner/work/mathledger/mathledger/config/curriculum_uplift_phase2.yaml")
    
    if not config_path.exists():
        pytest.skip("Phase II curriculum config not found")
    
    loader = CurriculumLoader(config_path)
    
    # Test loading each documented slice
    expected_slices = [
        "slice_uplift_goal",
        "slice_uplift_sparse",
        "slice_uplift_tree",
        "slice_uplift_dependency",
    ]
    
    for slice_name in expected_slices:
        try:
            items = loader.load_for_slice(slice_name)
            assert len(items) > 0, f"Slice {slice_name} has no items"
            
            # Verify deterministic ordering by reloading
            items2 = loader.load_for_slice(slice_name)
            assert len(items) == len(items2)
            for i, (item1, item2) in enumerate(zip(items, items2)):
                assert item1.formula == item2.formula, f"Order changed at {i}"
        except CurriculumNotFoundError:
            # Slice might not exist yet in config
            pass


@pytest.mark.unit
def test_curriculum_item_repr():
    """Test CurriculumItem string representation."""
    item = CurriculumItem(formula="p->q", hash="abc123")
    repr_str = repr(item)
    assert "p->q" in repr_str
    assert "abc123" in repr_str
