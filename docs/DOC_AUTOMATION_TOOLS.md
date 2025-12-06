# Documentation Automation Tools

This guide describes the documentation automation tools available in the MathLedger repository to maintain consistency, accuracy, and phase boundary clarity.

## Overview

Three automation tools are available:

1. **Snippet Check** - Validates CLI examples in documentation
2. **Evidence Pack TOC Generator** - Creates table of contents for evidence pack documents
3. **Phase Marker Lint** - Ensures Phase I/II boundaries are clearly marked and uplift disclaimers are present

## 1. Snippet Check - CLI Example Validator

### Purpose

Ensures that CLI examples in documentation are always executable and up-to-date by checking that referenced scripts and modules exist.

### Location

- Tool: `docs/snippet_check.py`
- Tests: `tests/docs/test_snippet_check.py`

### Usage

```bash
# Check all PHASE2*.md files (default)
python docs/snippet_check.py

# Check specific pattern
python docs/snippet_check.py --docs "docs/**/*.md"

# Verbose mode
python docs/snippet_check.py --verbose
```

### Features

- Extracts bash code blocks from markdown files
- Validates commands starting with:
  - `python experiments/run_uplift_*.py`
  - `python -m module.path`
  - `uv run python ...`
- Checks that referenced scripts/modules exist
- Validates basic syntax (e.g., matched quotes)
- Supports `# DOCTEST: SKIP` marker to skip validation of specific snippets

### Example: Skipping Future Scripts

If a documentation snippet references a script that will be created in the future:

```markdown
## Future Experiment

```bash
# DOCTEST: SKIP - summarize_uplift.py to be created in Phase II
uv run python experiments/summarize_uplift.py --baseline=results/baseline.jsonl
```
\```
```

### Exit Codes

- `0` - All CLI snippets are valid
- `1` - One or more snippets failed validation

### Integration

Add to pre-commit hooks or CI pipeline:

```yaml
# .github/workflows/docs-check.yml
- name: Validate CLI snippets
  run: python docs/snippet_check.py
```

## 2. Evidence Pack TOC Generator

### Purpose

Creates a machine- and human-readable table of contents describing all documents that constitute Evidence Pack v1.

### Location

- Tool: `docs/generate_evidence_pack_toc.py`
- Config: `docs/evidence_pack_config.yaml`
- Tests: `tests/docs/test_evidence_pack_toc.py`

### Usage

```bash
# Generate TOC (uses default config)
python docs/generate_evidence_pack_toc.py

# Use custom config
python docs/generate_evidence_pack_toc.py --config my_config.yaml

# Custom output directory
python docs/generate_evidence_pack_toc.py --output-dir artifacts/evidence
```

### Outputs

1. **`evidence_pack_v1_toc.json`** - Machine-readable JSON index
2. **`evidence_pack_v1_toc.md`** - Human-readable Markdown index

### Configuration Format

The config file (`evidence_pack_config.yaml`) uses this structure:

```yaml
metadata:
  version: "1.0"
  description: "Evidence Pack v1 - Phase I RFL Infrastructure"
  date: "2025-12"

documents:
  - path: "docs/PHASE2_RFL_UPLIFT_PLAN.md"
    description: "Phase II uplift experiment plan"
    phase: "PHASE II"
    category: "experiment-plan"
    tags: ["phase-ii", "uplift", "rfl"]
  
  - path: "docs/ARCHITECTURE.md"
    description: "System architecture documentation"
    phase: "General"
    category: "architecture"
    tags: ["architecture", "design"]
```

### Generated TOC Structure

The markdown TOC groups documents by phase and includes:

- Document path
- Description
- Category
- File size
- Tags (if provided)

### Adding New Documents

To add a document to the evidence pack:

1. Add entry to `docs/evidence_pack_config.yaml`
2. Run `python docs/generate_evidence_pack_toc.py`
3. Commit the updated TOC files

## 3. Phase Marker Lint - Consistency Checker

### Purpose

Ensures every relevant document clearly indicates Phase I vs Phase II and never accidentally claims uplift without appropriate disclaimers.

### Location

- Tool: `docs/phase_marker_lint.py`
- Config: `docs/phase_marker_rules.yaml`
- Tests: `tests/docs/test_phase_marker_lint.py`

### Usage

```bash
# Check all configured documents
python docs/phase_marker_lint.py

# Use custom config
python docs/phase_marker_lint.py --config my_rules.yaml

# Verbose mode
python docs/phase_marker_lint.py --verbose
```

### Features

- Checks for required Phase markers (e.g., "PHASE II")
- Validates uplift disclaimers are present when uplift is mentioned
- Detects forbidden claims (e.g., "uplift has been demonstrated" without disclaimers)
- Checks for required sections (optional)

### Configuration Format

The rules file (`docs/phase_marker_rules.yaml`) defines requirements per file pattern:

```yaml
rules:
  "docs/PHASE2*.md":
    phase_marker: "PHASE II"
    require_uplift_disclaimer: true
    forbidden_claims:
      - "uplift has been demonstrated"
      - "we have shown uplift"
    required_sections:
      - "Overview"
  
  "docs/RFL_PHASE_I_TRUTH_SOURCE.md":
    forbidden_claims:
      - "demonstrated uplift"
      - "showed improvement over baseline"
```

### Phase Marker Formats

The linter recognizes various phase marker formats:

```markdown
**STATUS: PHASE II**

> **STATUS: PHASE II — NOT YET RUN**

# PHASE II - Planning Document

PHASE II — Experimental Design
```

### Uplift Disclaimer Formats

The linter recognizes various disclaimer formats:

```markdown
NO UPLIFT CLAIMS MAY BE MADE

no empirical uplift has been demonstrated yet

no uplift has been demonstrated

uplift has not been demonstrated

NOT YET RUN

NOT RUN IN PHASE I
```

### Exit Codes

- `0` - All phase marker checks passed
- `1` - One or more checks failed

### Integration

Add to pre-commit hooks or CI pipeline:

```yaml
# .github/workflows/docs-check.yml
- name: Validate phase markers
  run: python docs/phase_marker_lint.py
```

## Best Practices

### For Documentation Authors

1. **Always mark Phase II documents** with status header:
   ```markdown
   > **STATUS: PHASE II — NOT YET RUN**
   > 
   > **Note:** No empirical uplift has been demonstrated yet.
   ```

2. **Use DOCTEST: SKIP** for future scripts in CLI examples:
   ```bash
   # DOCTEST: SKIP - script to be created in Phase II
   python experiments/future_script.py
   ```

3. **Run all three tools** before committing documentation:
   ```bash
   python docs/snippet_check.py
   python docs/phase_marker_lint.py
   python docs/generate_evidence_pack_toc.py
   ```

### For CI/CD Integration

Add all three tools to your CI pipeline:

```yaml
name: Documentation Checks

on: [pull_request]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Validate CLI snippets
        run: python docs/snippet_check.py
      
      - name: Validate phase markers
        run: python docs/phase_marker_lint.py
      
      - name: Generate evidence pack TOC
        run: python docs/generate_evidence_pack_toc.py
```

### For Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: snippet-check
        name: Validate CLI snippets
        entry: python docs/snippet_check.py
        language: python
        pass_filenames: false
      
      - id: phase-marker-lint
        name: Validate phase markers
        entry: python docs/phase_marker_lint.py
        language: python
        pass_filenames: false
```

## Testing

All tools have comprehensive test suites:

```bash
# Run all documentation tool tests
python -m pytest tests/docs/ -v

# Run specific test suite
python -m pytest tests/docs/test_snippet_check.py -v
python -m pytest tests/docs/test_evidence_pack_toc.py -v
python -m pytest tests/docs/test_phase_marker_lint.py -v
```

## Troubleshooting

### Snippet Check Issues

**Problem:** False positive on a valid command
```
Solution: Add # DOCTEST: SKIP marker to skip validation
```

**Problem:** Script exists but not found
```
Solution: Check that script path is relative to repository root
```

### Phase Marker Issues

**Problem:** Disclaimer not recognized
```
Solution: Check docs/phase_marker_lint.py for recognized disclaimer patterns
```

**Problem:** Too many false positives on forbidden claims
```
Solution: Refine forbidden_claims in docs/phase_marker_rules.yaml to be more specific
```

### Evidence Pack TOC Issues

**Problem:** Document not appearing in TOC
```
Solution: Check that file exists and path in config is correct (relative to repo root)
```

## Summary

These three tools work together to ensure documentation quality:

- **Snippet Check** ensures examples are executable
- **Evidence Pack TOC** provides a canonical index of all evidence documents
- **Phase Marker Lint** enforces Phase I/II boundary discipline and prevents accidental uplift claims

Run all three regularly to maintain high-quality, trustworthy documentation.
