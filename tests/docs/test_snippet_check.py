#!/usr/bin/env python3
"""
Tests for docs/snippet_check.py - CLI snippet validator.

Tests:
- Detection of bash code blocks
- Extraction of commands
- Validation of script references
- DOCTEST: SKIP marker support
- Multi-line command handling
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from docs.snippet_check import SnippetChecker


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure for testing."""
    # Create experiment script
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    (experiments_dir / "run_uplift_u2.py").write_text("# Test script\n")
    
    # Create module structure
    module_dir = tmp_path / "backend" / "tools"
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text("")
    (module_dir / "db_stats.py").write_text("# DB stats module\n")
    
    # Create docs directory
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    return tmp_path


def test_extract_bash_blocks(temp_project):
    """Test extraction of bash code blocks from markdown."""
    checker = SnippetChecker(temp_project)
    
    content = """
# Documentation

Some text here.

```bash
python experiments/run_uplift_u2.py --help
```

More text.

```python
# This should be ignored
print("hello")
```

```bash
echo "another bash block"
```
"""
    
    blocks = checker.extract_bash_blocks(content, "test.md")
    assert len(blocks) == 2
    assert "python experiments/run_uplift_u2.py" in blocks[0][1]
    assert "echo" in blocks[1][1]


def test_skip_marker(temp_project):
    """Test that DOCTEST: SKIP marker is respected."""
    checker = SnippetChecker(temp_project)
    
    code_with_skip = """
# DOCTEST: SKIP
python nonexistent_script.py
"""
    
    assert checker.should_skip_snippet(code_with_skip) is True
    
    code_without_skip = """
python experiments/run_uplift_u2.py
"""
    
    assert checker.should_skip_snippet(code_without_skip) is False


def test_extract_commands_simple(temp_project):
    """Test extraction of simple commands."""
    checker = SnippetChecker(temp_project)
    
    code = """
# Comment should be ignored
python experiments/run_uplift_u2.py --help

echo "test"
"""
    
    commands = checker.extract_commands(code)
    assert len(commands) == 2
    assert "python experiments/run_uplift_u2.py --help" in commands
    assert "echo" in commands[1]


def test_extract_commands_multiline(temp_project):
    """Test extraction of multi-line commands with backslash continuation."""
    checker = SnippetChecker(temp_project)
    
    code = """
python experiments/run_uplift_u2.py \\
  --slice-name=slice_uplift_goal \\
  --mode=baseline \\
  --cycles=500
"""
    
    commands = checker.extract_commands(code)
    assert len(commands) == 1
    assert "--slice-name=slice_uplift_goal" in commands[0]
    assert "--mode=baseline" in commands[0]
    assert "--cycles=500" in commands[0]


def test_is_relevant_command(temp_project):
    """Test detection of relevant Python CLI commands."""
    checker = SnippetChecker(temp_project)
    
    # Relevant commands
    assert checker.is_relevant_command("python experiments/run_uplift_u2.py") is True
    assert checker.is_relevant_command("python -m backend.tools.db_stats") is True
    assert checker.is_relevant_command("uv run python experiments/run_uplift_u2.py") is True
    assert checker.is_relevant_command("uv run python -m pytest") is True
    
    # Irrelevant commands
    assert checker.is_relevant_command("echo 'hello'") is False
    assert checker.is_relevant_command("ls -la") is False
    assert checker.is_relevant_command("make test") is False


def test_validate_existing_script(temp_project):
    """Test validation of existing script reference."""
    checker = SnippetChecker(temp_project)
    
    cmd = "python experiments/run_uplift_u2.py --help"
    result = checker.validate_command(cmd, "test.md", 10)
    
    assert result is True
    assert len(checker.errors) == 0


def test_validate_missing_script(temp_project):
    """Test detection of missing script."""
    checker = SnippetChecker(temp_project)
    
    cmd = "python experiments/nonexistent.py --help"
    result = checker.validate_command(cmd, "test.md", 10)
    
    assert result is False
    assert len(checker.errors) == 1
    assert "nonexistent.py" in checker.errors[0]


def test_validate_existing_module(temp_project):
    """Test validation of existing module reference."""
    checker = SnippetChecker(temp_project)
    
    cmd = "python -m backend.tools.db_stats"
    result = checker.validate_command(cmd, "test.md", 15)
    
    assert result is True
    assert len(checker.errors) == 0


def test_validate_missing_module(temp_project):
    """Test detection of missing module."""
    checker = SnippetChecker(temp_project)
    
    cmd = "python -m nonexistent.module"
    result = checker.validate_command(cmd, "test.md", 15)
    
    assert result is False
    assert len(checker.errors) == 1
    assert "nonexistent.module" in checker.errors[0]


def test_validate_uv_run_prefix(temp_project):
    """Test that uv run prefix is handled correctly."""
    checker = SnippetChecker(temp_project)
    
    cmd = "uv run python experiments/run_uplift_u2.py --cycles=10"
    result = checker.validate_command(cmd, "test.md", 20)
    
    assert result is True
    assert len(checker.errors) == 0


def test_check_file_with_valid_snippets(temp_project):
    """Test checking a file with valid CLI snippets."""
    docs_dir = temp_project / "docs"
    test_doc = docs_dir / "test_valid.md"
    
    test_doc.write_text("""
# Test Documentation

Example command:

```bash
python experiments/run_uplift_u2.py --help
```
""")
    
    checker = SnippetChecker(temp_project)
    result = checker.check_file(test_doc)
    
    assert result is True
    assert len(checker.errors) == 0


def test_check_file_with_invalid_snippets(temp_project):
    """Test checking a file with invalid CLI snippets."""
    docs_dir = temp_project / "docs"
    test_doc = docs_dir / "test_invalid.md"
    
    test_doc.write_text("""
# Test Documentation

Example command:

```bash
python experiments/missing_script.py --help
```
""")
    
    checker = SnippetChecker(temp_project)
    result = checker.check_file(test_doc)
    
    assert result is False
    assert len(checker.errors) == 1
    assert "missing_script.py" in checker.errors[0]


def test_check_file_with_skip_marker(temp_project):
    """Test that files with SKIP marker don't produce errors."""
    docs_dir = temp_project / "docs"
    test_doc = docs_dir / "test_skip.md"
    
    test_doc.write_text("""
# Test Documentation

Example command (should be skipped):

```bash
# DOCTEST: SKIP
python experiments/missing_script.py --help
```
""")
    
    checker = SnippetChecker(temp_project)
    result = checker.check_file(test_doc)
    
    assert result is True
    assert len(checker.errors) == 0


def test_unmatched_quotes_detection(temp_project):
    """Test detection of unmatched quotes in commands."""
    checker = SnippetChecker(temp_project)
    
    cmd = 'python experiments/run_uplift_u2.py --name="test'
    result = checker.validate_command(cmd, "test.md", 25)
    
    assert result is False
    assert len(checker.errors) == 1
    assert "Unmatched quote" in checker.errors[0]


def test_multiple_errors_accumulation(temp_project):
    """Test that multiple errors are accumulated."""
    docs_dir = temp_project / "docs"
    test_doc = docs_dir / "test_multi_error.md"
    
    test_doc.write_text("""
# Test Documentation

```bash
python experiments/missing1.py
python experiments/missing2.py
```
""")
    
    checker = SnippetChecker(temp_project)
    checker.check_file(test_doc)
    
    assert len(checker.errors) == 2
    assert "missing1.py" in checker.errors[0]
    assert "missing2.py" in checker.errors[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
