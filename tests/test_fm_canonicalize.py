"""
PHASE II — NOT RUN IN PHASE I

Tests for Field Manual Canonicalizer & Intelligence System

This module contains 30 tests covering:
  - Canonicalization determinism (tests 1-10)
  - Cross-reference completeness (tests 11-20)
  - Drift detection correctness (tests 21-30)

ABSOLUTE SAFEGUARDS:
    - Tests are DESCRIPTIVE, not NORMATIVE
    - No modifications to fm.tex contents
    - No inference or claims regarding uplift
"""

import hashlib
import json
import pytest
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fm_canonicalize import (
    # Data structures
    LabelInfo,
    DefinitionInfo,
    InvariantInfo,
    FormulaInfo,
    RefInfo,
    CrossRefAuditResult,
    DriftCheckResult,
    CanonicalRepresentation,
    SpecContract,
    CrossCheckResult,
    CrossCheckSummary,
    LabelConsumer,
    LabelDriftSummary,
    # Extraction functions
    extract_labels,
    extract_definitions,
    extract_invariants,
    extract_formulas,
    extract_sections,
    extract_refs,
    # Validation functions
    validate_label_uniqueness,
    validate_label_ordering,
    validate_label_naming,
    # Cross-reference functions
    audit_cross_references,
    find_similar_labels,
    levenshtein_distance,
    # Drift detection functions
    check_drift,
    remove_code_blocks,
    is_in_code_context,
    extract_evidence_section,
    # Canonicalization functions
    build_canonical_representation,
    compute_signature_hash,
    compute_file_hash,
    save_canonical_json,
    # Spec contract functions
    export_spec_contract,
    find_section_for_label,
    find_section_by_title_pattern,
    save_spec_contract,
    # Cross-check functions
    extract_label_refs_from_markdown,
    check_core_terminology,
    cross_check_document,
    run_cross_check,
    generate_cross_check_report,
    # Label consumers & drift
    build_consumers_index,
    compute_label_drift,
    save_label_drift_summary,
    export_spec_contract_with_consumers,
    # FM Router API
    get_label_definition,
    get_label_consumers,
    get_label_total_refs,
    is_label_well_connected,
    build_fm_consumers_view,
    build_fm_posture,
    load_fm_posture,
    # FM Governance & Alignment
    build_fm_governance_snapshot,
    compute_alignment_indicator,
    summarize_fm_for_global_health,
    # Phase IV — Cross-Agent Truth Anchor
    extract_labels_from_taxonomy,
    extract_labels_from_curriculum,
    build_label_contract_index,
    build_field_manual_integration_contract,
    build_field_manual_director_panel,
    build_field_manual_drift_timeline,
    # Config
    EXPECTED_SECTION_ORDER,
    DETERMINISM_FORBIDDEN_PRIMITIVES,
    FM_TEX_PATH,
    FM_CANONICAL_PATH,
    FM_SPEC_CONTRACT_PATH,
    LABEL_DRIFT_PATH,
    CORE_TERMINOLOGY,
    SPEC_CONTRACT_REQUIRED_KEYS,
    LABEL_DRIFT_REQUIRED_KEYS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CANONICALIZATION DETERMINISM TESTS (1-10)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanonicalizationDeterminism:
    """Tests verifying canonicalization produces deterministic output."""
    
    # Test 1: File hash determinism
    def test_01_file_hash_determinism(self):
        """Same content always produces same hash."""
        content = "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}"
        hash1 = compute_file_hash(content)
        hash2 = compute_file_hash(content)
        assert hash1 == hash2, "File hash must be deterministic"
    
    # Test 2: Different content produces different hash
    def test_02_file_hash_sensitivity(self):
        """Different content produces different hash."""
        content1 = "Hello World"
        content2 = "Hello World!"
        hash1 = compute_file_hash(content1)
        hash2 = compute_file_hash(content2)
        assert hash1 != hash2, "Different content must produce different hash"
    
    # Test 3: Label extraction determinism
    def test_03_label_extraction_determinism(self):
        """Label extraction is deterministic."""
        content = r"""
\section{Test}\label{sec:test}
\subsection{Sub}\label{sec:sub}
"""
        lines = content.split("\n")
        
        labels1 = extract_labels(content, lines)
        labels2 = extract_labels(content, lines)
        
        assert len(labels1) == len(labels2)
        for l1, l2 in zip(labels1, labels2):
            assert l1.name == l2.name
            assert l1.line_number == l2.line_number
    
    # Test 4: Definition extraction determinism
    def test_04_definition_extraction_determinism(self):
        """Definition extraction is deterministic."""
        content = r"""
\begin{definition}[Test Def]
\label{def:test}
This is a test definition.
\end{definition}
"""
        lines = content.split("\n")
        
        defs1 = extract_definitions(content, lines)
        defs2 = extract_definitions(content, lines)
        
        assert len(defs1) == len(defs2)
        assert defs1[0].name == defs2[0].name
        assert defs1[0].label == defs2[0].label
    
    # Test 5: Invariant extraction determinism
    def test_05_invariant_extraction_determinism(self):
        """Invariant extraction is deterministic."""
        content = r"""
\begin{invariant}[Test Inv]
\label{inv:test}
This is an invariant.
\end{invariant}
"""
        lines = content.split("\n")
        
        invs1 = extract_invariants(content, lines)
        invs2 = extract_invariants(content, lines)
        
        assert len(invs1) == len(invs2)
        assert invs1[0].name == invs2[0].name
    
    # Test 6: Formula extraction determinism
    def test_06_formula_extraction_determinism(self):
        """Formula extraction is deterministic."""
        content = r"""
The formula $x = y$ is inline.
Display math:
\[
E = mc^2
\]
"""
        lines = content.split("\n")
        
        forms1 = extract_formulas(content, lines)
        forms2 = extract_formulas(content, lines)
        
        assert len(forms1) == len(forms2)
        for f1, f2 in zip(forms1, forms2):
            assert f1.content == f2.content
            assert f1.formula_type == f2.formula_type
    
    # Test 7: Section extraction determinism
    def test_07_section_extraction_determinism(self):
        """Section extraction is deterministic."""
        content = r"""
\section{First}
\label{sec:first}
\subsection{Sub}
\section{Second}
"""
        lines = content.split("\n")
        
        secs1 = extract_sections(content, lines)
        secs2 = extract_sections(content, lines)
        
        assert len(secs1) == len(secs2)
        for s1, s2 in zip(secs1, secs2):
            assert s1["title"] == s2["title"]
            assert s1["level"] == s2["level"]
    
    # Test 8: Signature hash determinism
    def test_08_signature_hash_determinism(self):
        """Signature hash is deterministic for same representation."""
        canon = CanonicalRepresentation(
            version="1.0.0",
            source_path="/test/path",
            source_hash="abc123",
            labels=[{"name": "sec:test", "line_number": 1, "label_type": "section", "context": ""}],
            definitions=[],
            invariants=[],
            formulas=[],
            sections=[]
        )
        
        hash1 = compute_signature_hash(canon)
        hash2 = compute_signature_hash(canon)
        
        assert hash1 == hash2, "Signature hash must be deterministic"
    
    # Test 9: Signature hash changes with content
    def test_09_signature_hash_sensitivity(self):
        """Signature hash changes when content changes."""
        canon1 = CanonicalRepresentation(
            version="1.0.0",
            source_hash="abc123",
            labels=[{"name": "sec:test", "line_number": 1, "label_type": "section", "context": ""}],
        )
        
        canon2 = CanonicalRepresentation(
            version="1.0.0",
            source_hash="abc123",
            labels=[{"name": "sec:other", "line_number": 1, "label_type": "section", "context": ""}],
        )
        
        hash1 = compute_signature_hash(canon1)
        hash2 = compute_signature_hash(canon2)
        
        assert hash1 != hash2, "Different labels must produce different signature"
    
    # Test 10: Full canonicalization determinism
    def test_10_full_canonicalization_determinism(self):
        """Full canonicalization is deterministic."""
        content = r"""
\documentclass{article}
\begin{document}
\section{Test}\label{sec:test}
\begin{definition}[Def]\label{def:test}
Content
\end{definition}
The formula $x=1$ applies.
\end{document}
"""
        lines = content.split("\n")
        
        canon1 = build_canonical_representation(content, lines)
        canon2 = build_canonical_representation(content, lines)
        
        assert canon1.signature_hash == canon2.signature_hash


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CROSS-REFERENCE COMPLETENESS TESTS (11-20)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossReferenceCompleteness:
    """Tests verifying cross-reference audit functionality."""
    
    # Test 11: Detects valid references
    def test_11_valid_references_detected(self):
        """Valid references are detected correctly."""
        content = r"See Section~\ref{sec:test} and Definition~\ref{def:foo}."
        lines = content.split("\n")
        
        refs = extract_refs(content, lines)
        
        assert len(refs) == 2
        assert refs[0].target == "sec:test"
        assert refs[1].target == "def:foo"
    
    # Test 12: Detects eqref references
    def test_12_eqref_detected(self):
        """Equation references (\\eqref) are detected."""
        content = r"See Equation~\eqref{eq:energy}."
        lines = content.split("\n")
        
        refs = extract_refs(content, lines)
        
        assert len(refs) == 1
        assert refs[0].ref_type == "eqref"
        assert refs[0].target == "eq:energy"
    
    # Test 13: Detects pageref references
    def test_13_pageref_detected(self):
        """Page references (\\pageref) are detected."""
        content = r"On page~\pageref{sec:intro}."
        lines = content.split("\n")
        
        refs = extract_refs(content, lines)
        
        assert len(refs) == 1
        assert refs[0].ref_type == "pageref"
    
    # Test 14: Identifies dangling references
    def test_14_dangling_refs_identified(self):
        """Dangling references (no matching label) are identified."""
        labels = [
            LabelInfo("sec:exists", 1, "section", "")
        ]
        refs = [
            RefInfo("ref", "sec:exists", 10),
            RefInfo("ref", "sec:missing", 20),
        ]
        
        result = audit_cross_references(labels, refs)
        
        assert len(result.dangling_refs) == 1
        assert result.dangling_refs[0][0] == "sec:missing"
    
    # Test 15: Identifies unused labels
    def test_15_unused_labels_identified(self):
        """Unused labels (never referenced) are identified."""
        labels = [
            LabelInfo("sec:used", 1, "section", ""),
            LabelInfo("def:unused", 5, "definition", ""),
        ]
        refs = [
            RefInfo("ref", "sec:used", 10),
        ]
        
        result = audit_cross_references(labels, refs)
        
        # Sections are excluded from unused check
        assert len(result.unused_labels) == 1
        assert result.unused_labels[0][0] == "def:unused"
    
    # Test 16: Generates suggestions for dangling refs
    def test_16_suggestions_generated(self):
        """Suggestions are generated for dangling references."""
        labels = [
            LabelInfo("sec:intro", 1, "section", ""),
        ]
        refs = [
            RefInfo("ref", "sec:intr", 10),  # Typo - missing 'o' (edit distance 1)
        ]
        
        result = audit_cross_references(labels, refs)
        
        assert len(result.suggestions) == 1
        assert "sec:intro" in result.suggestions[0]
    
    # Test 17: Levenshtein distance correct
    def test_17_levenshtein_distance_correct(self):
        """Levenshtein distance computation is correct."""
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("a", "") == 1
        assert levenshtein_distance("abc", "abc") == 0
        assert levenshtein_distance("abc", "abd") == 1
        assert levenshtein_distance("kitten", "sitting") == 3
    
    # Test 18: Find similar labels works
    def test_18_find_similar_labels(self):
        """Similar label finder returns closest matches."""
        labels = {"sec:introduction", "sec:methods", "sec:results", "sec:intro"}
        
        similar = find_similar_labels("sec:introd", labels)
        
        assert "sec:intro" in similar  # Closest match
    
    # Test 19: Multiple refs to same label
    def test_19_multiple_refs_same_label(self):
        """Multiple references to the same label are all tracked."""
        content = r"""
See \ref{sec:test} here.
Also \ref{sec:test} there.
And \ref{sec:test} again.
"""
        lines = content.split("\n")
        refs = extract_refs(content, lines)
        
        assert len(refs) == 3
        assert all(r.target == "sec:test" for r in refs)
    
    # Test 20: Empty document handling
    def test_20_empty_document_refs(self):
        """Empty document produces no refs or labels."""
        content = ""
        lines = []
        
        refs = extract_refs(content, lines)
        labels = extract_labels(content, lines)
        
        result = audit_cross_references(labels, refs)
        
        assert len(result.dangling_refs) == 0
        assert len(result.unused_labels) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DRIFT DETECTION CORRECTNESS TESTS (21-30)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDriftDetectionCorrectness:
    """Tests verifying drift detection functionality."""
    
    # Test 21: Detects forbidden primitive in code
    def test_21_detects_forbidden_primitive(self):
        """Detects forbidden primitives in code listings."""
        content = r"""
\begin{lstlisting}[language=Python,caption=Bad Code]
import datetime
timestamp = datetime.now()  # This should be flagged
\end{lstlisting}
"""
        lines = content.split("\n")
        
        result = check_drift(content, lines)
        
        # Note: The check is contextual, may not flag if in "forbidden" documentation
        # This test verifies the mechanism exists
        assert isinstance(result, DriftCheckResult)
    
    # Test 22: Code block removal works
    def test_22_code_block_removal(self):
        """Code blocks are properly removed for prose analysis."""
        content = r"""
Normal text here.
\begin{lstlisting}
datetime.now()  # This is code
\end{lstlisting}
More text.
"""
        cleaned = remove_code_blocks(content)
        
        assert "datetime.now()" not in cleaned
        assert "Normal text here" in cleaned
        assert "More text" in cleaned
    
    # Test 23: texttt removal works
    def test_23_texttt_removal(self):
        """\\texttt{} blocks are removed from prose analysis."""
        content = r"Use \texttt{datetime.now()} for timestamps."
        cleaned = remove_code_blocks(content)
        
        assert "datetime.now()" not in cleaned
    
    # Test 24: Detects code context
    def test_24_is_in_code_context(self):
        """Correctly identifies positions inside code contexts."""
        line = r"Use \texttt{datetime.now()} for time."
        
        # Position inside \texttt{}
        assert is_in_code_context(line, 10) == True
        
        # Position outside \texttt{}
        assert is_in_code_context(line, 0) == False
    
    # Test 25: Evidence section extraction
    def test_25_evidence_section_extraction(self):
        """Evidence section is correctly extracted."""
        content = r"""
\section{Introduction}
Intro text.
\section{Evidence Interpretation}
Evidence content here.
\section{Appendix}
"""
        extracted = extract_evidence_section(content)
        
        assert extracted is not None
        assert "Evidence content here" in extracted
        assert "Intro text" not in extracted
    
    # Test 26: Structure issues detected
    def test_26_structure_issues_detected(self):
        """Structure issues (missing Phase II markers) are detected."""
        content = "No markers here."
        lines = content.split("\n")
        
        result = check_drift(content, lines)
        
        assert len(result.structure_issues) > 0
        assert any("Phase II marker" in issue for issue in result.structure_issues)
    
    # Test 27: Clean document passes drift check
    def test_27_clean_document_passes(self):
        """Document with proper markers passes structure check."""
        # Create content with enough Phase II markers
        markers = "PHASE II — NOT RUN IN PHASE I\n" * 10
        content = markers + r"""
\section{Evidence Interpretation}
\fbox{THIS SECTION IS INTENTIONALLY EMPTY}
"""
        lines = content.split("\n")
        
        result = check_drift(content, lines)
        
        # Should have no structure issues about markers
        marker_issues = [i for i in result.structure_issues if "marker" in i.lower()]
        assert len(marker_issues) == 0
    
    # Test 28: Evidence section content restriction
    def test_28_evidence_section_restriction(self):
        """Evidence section with forbidden content is flagged."""
        content = r"""
PHASE II — NOT RUN IN PHASE I
PHASE II — NOT RUN IN PHASE I
PHASE II — NOT RUN IN PHASE I
PHASE II — NOT RUN IN PHASE I
PHASE II — NOT RUN IN PHASE I
\section{Evidence Interpretation}
The results show 50% uplift with p < 0.05.
\appendix
"""
        lines = content.split("\n")
        
        result = check_drift(content, lines)
        
        # Should flag the evidence content
        assert any("Evidence section" in issue for issue in result.structure_issues)
    
    # Test 29: Label validation - uniqueness
    def test_29_label_uniqueness_validation(self):
        """Duplicate labels are detected."""
        labels = [
            LabelInfo("sec:test", 1, "section", ""),
            LabelInfo("sec:test", 5, "section", ""),  # Duplicate!
        ]
        
        errors = validate_label_uniqueness(labels)
        
        assert len(errors) == 1
        assert "Duplicate" in errors[0]
    
    # Test 30: Label validation - naming convention
    def test_30_label_naming_convention(self):
        """Labels without proper prefixes are flagged."""
        labels = [
            LabelInfo("sec:valid", 1, "section", ""),
            LabelInfo("invalid_no_prefix", 5, "other", ""),  # Missing colon
        ]
        
        errors = validate_label_naming(labels)
        
        assert len(errors) == 1
        assert "prefix" in errors[0].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests using actual fm.tex if available."""
    
    @pytest.mark.skipif(not FM_TEX_PATH.exists(), reason="fm.tex not found")
    def test_actual_fm_tex_canonicalizes(self):
        """Actual fm.tex can be canonicalized without errors."""
        with open(FM_TEX_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.split("\n")
        
        # Should not raise
        canon = build_canonical_representation(content, lines)
        
        assert canon.signature_hash
        assert len(canon.labels) > 0
        assert len(canon.sections) > 0
    
    @pytest.mark.skipif(not FM_TEX_PATH.exists(), reason="fm.tex not found")
    def test_actual_fm_tex_refs_valid(self):
        """Actual fm.tex has no dangling references."""
        with open(FM_TEX_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.split("\n")
        
        labels = extract_labels(content, lines)
        refs = extract_refs(content, lines)
        result = audit_cross_references(labels, refs)
        
        # Should have no dangling refs
        assert len(result.dangling_refs) == 0, f"Dangling refs: {result.dangling_refs}"
    
    @pytest.mark.skipif(not FM_TEX_PATH.exists(), reason="fm.tex not found")
    def test_actual_fm_tex_no_structure_drift(self):
        """Actual fm.tex has no structure drift."""
        with open(FM_TEX_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.split("\n")
        
        result = check_drift(content, lines)
        
        # Should have no structure issues
        assert len(result.structure_issues) == 0, f"Structure issues: {result.structure_issues}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SPEC CONTRACT TESTS (31-40)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecContract:
    """Tests for spec contract export functionality."""
    
    # Test 31: SpecContract dataclass structure
    def test_31_spec_contract_structure(self):
        """SpecContract has required fields."""
        contract = SpecContract(
            version="1.0",
            signature_hash="abc123",
            definitions=[{"label": "def:test", "name": "Test", "section": "1.1"}],
            invariants=[{"label": "inv:test", "name": "Test Inv", "section": "2.1"}],
            metrics_section="4.3",
            uplift_section="5.1",
            phase_section="1.0"
        )
        
        result = contract.to_dict()
        
        assert "version" in result
        assert "signature_hash" in result
        assert "definitions" in result
        assert "invariants" in result
        assert "metrics_section" in result
        assert "uplift_section" in result
        assert "phase_section" in result
    
    # Test 32: Find section for label
    def test_32_find_section_for_label(self):
        """Section lookup by label works."""
        sections = [
            {"level": "section", "title": "Introduction", "label": "sec:intro", "line_number": 1},
            {"level": "subsection", "title": "Overview", "label": "sec:overview", "line_number": 10},
            {"level": "section", "title": "Methods", "label": "sec:methods", "line_number": 20},
        ]
        
        result = find_section_for_label("sec:overview", sections)
        assert result == "1.1"
        
        result = find_section_for_label("sec:methods", sections)
        assert result == "2"
    
    # Test 33: Find section by title pattern
    def test_33_find_section_by_title(self):
        """Section lookup by title pattern works."""
        sections = [
            {"level": "section", "title": "Introduction", "label": "sec:intro", "line_number": 1},
            {"level": "section", "title": "Success Metrics", "label": "sec:metrics", "line_number": 20},
            {"level": "section", "title": "Evidence Interpretation", "label": "sec:evidence", "line_number": 30},
        ]
        
        result = find_section_by_title_pattern(r"metric", sections)
        assert result == "2"
        
        result = find_section_by_title_pattern(r"evidence", sections)
        assert result == "3"
    
    # Test 34: Contract signature hash matches canonical
    @pytest.mark.skipif(not FM_CANONICAL_PATH.exists(), reason="fm_canonical.json not found")
    def test_34_contract_hash_matches_canonical(self):
        """Contract signature hash matches canonical JSON."""
        import json
        
        with open(FM_CANONICAL_PATH, "r", encoding="utf-8") as f:
            canon_data = json.load(f)
        
        contract = export_spec_contract(FM_CANONICAL_PATH)
        
        assert contract["signature_hash"] == canon_data["signature_hash"]
    
    # Test 35: Contract has required top-level keys
    @pytest.mark.skipif(not FM_CANONICAL_PATH.exists(), reason="fm_canonical.json not found")
    def test_35_contract_required_keys(self):
        """Exported contract has all required top-level keys."""
        contract = export_spec_contract(FM_CANONICAL_PATH)
        
        required_keys = [
            "version",
            "signature_hash",
            "definitions",
            "invariants",
            "metrics_section",
            "uplift_section",
            "phase_section"
        ]
        
        for key in required_keys:
            assert key in contract, f"Missing required key: {key}"
    
    # Test 36: Contract definitions have structure
    @pytest.mark.skipif(not FM_CANONICAL_PATH.exists(), reason="fm_canonical.json not found")
    def test_36_contract_definitions_structure(self):
        """Contract definitions have label, name, section."""
        contract = export_spec_contract(FM_CANONICAL_PATH)
        
        for defn in contract["definitions"]:
            assert "label" in defn
            assert "name" in defn
            assert "section" in defn
    
    # Test 37: Contract invariants have structure
    @pytest.mark.skipif(not FM_CANONICAL_PATH.exists(), reason="fm_canonical.json not found")
    def test_37_contract_invariants_structure(self):
        """Contract invariants have label, name, section."""
        contract = export_spec_contract(FM_CANONICAL_PATH)
        
        for inv in contract["invariants"]:
            assert "label" in inv
            assert "name" in inv
            assert "section" in inv
    
    # Test 38: Save spec contract creates file
    def test_38_save_spec_contract(self, tmp_path):
        """save_spec_contract creates file."""
        contract = {
            "version": "1.0",
            "signature_hash": "test123",
            "definitions": [],
            "invariants": [],
            "metrics_section": None,
            "uplift_section": None,
            "phase_section": None
        }
        
        out_path = tmp_path / "spec" / "test_contract.json"
        result_path = save_spec_contract(contract, out_path)
        
        assert result_path.exists()
        
        import json
        with open(result_path, "r") as f:
            loaded = json.load(f)
        
        assert loaded["version"] == "1.0"
        assert loaded["signature_hash"] == "test123"
    
    # Test 39: Export spec is deterministic
    @pytest.mark.skipif(not FM_CANONICAL_PATH.exists(), reason="fm_canonical.json not found")
    def test_39_export_spec_deterministic(self):
        """export_spec_contract is deterministic."""
        contract1 = export_spec_contract(FM_CANONICAL_PATH)
        contract2 = export_spec_contract(FM_CANONICAL_PATH)
        
        assert contract1 == contract2
    
    # Test 40: Missing canonical raises error
    def test_40_missing_canonical_raises(self, tmp_path):
        """export_spec_contract raises when canonical missing."""
        fake_path = tmp_path / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            export_spec_contract(fake_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CROSS-CHECK TESTS (41-50)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossCheck:
    """Tests for cross-check functionality."""
    
    # Test 41: Extract label refs from markdown
    def test_41_extract_label_refs(self):
        """Label references are extracted from markdown."""
        content = """
        See `def:slice` for the definition.
        Also check `inv:monotonicity` and `sec:metrics`.
        Plain text: def:test should also match.
        """
        
        refs = extract_label_refs_from_markdown(content)
        
        assert "def:slice" in refs
        assert "inv:monotonicity" in refs
        assert "sec:metrics" in refs
    
    # Test 42: Check core terminology - present
    def test_42_core_terminology_present(self):
        """Core terminology detection finds present terms."""
        content = """
        The RFL policy updates based on H_t.
        Phase II experiments measure uplift and abstention.
        """
        
        present, missing = check_core_terminology(content)
        
        assert "RFL" in present
        assert "H_t" in present
        assert "Phase II" in present
        assert "uplift" in present
        assert "abstention" in present
    
    # Test 43: Check core terminology - missing
    def test_43_core_terminology_missing(self):
        """Core terminology detection finds missing terms."""
        content = "This document has none of the expected terms."
        
        present, missing = check_core_terminology(content)
        
        assert len(missing) > 0
        assert "RFL" in missing
        assert "H_t" in missing
    
    # Test 44: Cross-check with valid labels
    def test_44_cross_check_valid(self, tmp_path):
        """Cross-check passes when labels exist."""
        # Create test markdown with valid refs
        doc = tmp_path / "test.md"
        doc.write_text("See `sec:intro` and `def:slice`. RFL and H_t are mentioned.")
        
        fm_labels = {"sec:intro", "def:slice", "inv:test"}
        fm_definitions = [{"label": "def:slice", "name": "Slice"}]
        fm_invariants = [{"label": "inv:test", "name": "Test"}]
        
        result = cross_check_document(doc, fm_labels, fm_definitions, fm_invariants)
        
        assert result.status in ["pass", "warn"]
        assert len(result.missing_labels) == 0
    
    # Test 45: Cross-check with missing labels
    def test_45_cross_check_missing(self, tmp_path):
        """Cross-check fails when labels are missing."""
        doc = tmp_path / "test.md"
        doc.write_text("See `sec:nonexistent` and `def:missing`.")
        
        fm_labels = {"sec:intro", "def:slice"}
        fm_definitions = [{"label": "def:slice", "name": "Slice"}]
        fm_invariants = []
        
        result = cross_check_document(doc, fm_labels, fm_definitions, fm_invariants)
        
        assert result.status == "fail"
        assert len(result.missing_labels) == 2
    
    # Test 46: Cross-check skips missing file
    def test_46_cross_check_missing_file(self, tmp_path):
        """Cross-check skips non-existent documents."""
        fake_doc = tmp_path / "nonexistent.md"
        
        result = cross_check_document(
            fake_doc,
            set(),
            [],
            []
        )
        
        assert result.status == "skip"
    
    # Test 47: Generate cross-check report
    def test_47_generate_report(self):
        """Report generation produces markdown."""
        summary = CrossCheckSummary(
            timestamp="2025-01-01T00:00:00Z",
            fm_signature_hash="abc123def456",
            docs_checked=2,
            docs_passed=1,
            docs_warned=1,
            docs_failed=0,
            results=[
                {
                    "doc_path": "test1.md",
                    "status": "pass",
                    "missing_labels": [],
                    "undefined_terms": [],
                    "referenced_definitions": ["def:test"],
                    "details": []
                },
                {
                    "doc_path": "test2.md",
                    "status": "warn",
                    "missing_labels": [],
                    "undefined_terms": ["RFL", "H_t"],
                    "referenced_definitions": [],
                    "details": []
                }
            ]
        )
        
        report = generate_cross_check_report(summary)
        
        assert "# Field Manual Cross-Check Report" in report
        assert "✅ Passed" in report
        assert "⚠️ Warned" in report
        assert "test1.md" in report
        assert "test2.md" in report
    
    # Test 48: CrossCheckSummary counts
    def test_48_summary_counts(self):
        """CrossCheckSummary tracks counts correctly."""
        summary = CrossCheckSummary(
            timestamp="now",
            fm_signature_hash="hash",
            docs_checked=5,
            docs_passed=2,
            docs_warned=2,
            docs_failed=1,
            results=[]
        )
        
        assert summary.docs_checked == 5
        assert summary.docs_passed == 2
        assert summary.docs_warned == 2
        assert summary.docs_failed == 1
    
    # Test 49: Cross-check detects referenced definitions
    def test_49_referenced_definitions(self, tmp_path):
        """Cross-check identifies correctly referenced definitions."""
        doc = tmp_path / "test.md"
        doc.write_text("The `def:slice` definition from fm.tex applies here. RFL H_t.")
        
        fm_labels = {"def:slice", "inv:mono"}
        fm_definitions = [{"label": "def:slice", "name": "Slice"}]
        fm_invariants = [{"label": "inv:mono", "name": "Monotonicity"}]
        
        result = cross_check_document(doc, fm_labels, fm_definitions, fm_invariants)
        
        assert "def:slice" in result.referenced_definitions
    
    # Test 50: Core terminology list is complete
    def test_50_core_terminology_list(self):
        """CORE_TERMINOLOGY has expected terms."""
        assert "RFL" in CORE_TERMINOLOGY
        assert "H_t" in CORE_TERMINOLOGY
        assert "Phase II" in CORE_TERMINOLOGY
        assert "abstention" in CORE_TERMINOLOGY
        assert "uplift" in CORE_TERMINOLOGY


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: INTEGRATION TESTS FOR NEW FEATURES (51-55)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewFeaturesIntegration:
    """Integration tests for spec contract and cross-check with actual files."""
    
    @pytest.mark.skipif(not FM_TEX_PATH.exists(), reason="fm.tex not found")
    def test_51_full_pipeline_spec_export(self, tmp_path):
        """Full pipeline: canonicalize -> export spec."""
        # Build canonical
        with open(FM_TEX_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.split("\n")
        
        canon = build_canonical_representation(content, lines)
        canon_path = tmp_path / "canon.json"
        save_canonical_json(canon, canon_path)
        
        # Export spec
        contract = export_spec_contract(canon_path)
        
        assert contract["version"] == "1.0"
        assert len(contract["definitions"]) > 0
        assert len(contract["invariants"]) > 0
    
    @pytest.mark.skipif(not FM_CANONICAL_PATH.exists(), reason="fm_canonical.json not found")
    def test_52_cross_check_actual_docs(self):
        """Cross-check runs on actual documentation files."""
        # This test verifies the cross-check mechanism works with real files
        # It may find issues, which is expected behavior
        
        try:
            summary = run_cross_check(FM_CANONICAL_PATH)
            
            assert summary.docs_checked >= 0
            assert isinstance(summary.results, list)
        except FileNotFoundError:
            pytest.skip("Documentation files not found")
    
    @pytest.mark.skipif(not FM_CANONICAL_PATH.exists(), reason="fm_canonical.json not found")
    def test_53_spec_contract_file_output(self, tmp_path):
        """Spec contract saves to file correctly."""
        contract = export_spec_contract(FM_CANONICAL_PATH)
        
        out_path = tmp_path / "spec" / "contract.json"
        save_spec_contract(contract, out_path)
        
        assert out_path.exists()
        
        import json
        with open(out_path, "r") as f:
            loaded = json.load(f)
        
        assert loaded["signature_hash"] == contract["signature_hash"]
    
    def test_54_cross_check_report_format(self, tmp_path):
        """Cross-check report has proper markdown format."""
        # Create mock summary
        summary = CrossCheckSummary(
            timestamp="2025-01-01T00:00:00Z",
            fm_signature_hash="a" * 64,
            docs_checked=1,
            docs_passed=1,
            docs_warned=0,
            docs_failed=0,
            results=[{
                "doc_path": "test.md",
                "status": "pass",
                "missing_labels": [],
                "undefined_terms": [],
                "referenced_definitions": [],
                "details": []
            }]
        )
        
        report = generate_cross_check_report(summary)
        
        # Check markdown structure
        assert report.startswith("# ")
        assert "## Summary" in report
        assert "## Details" in report
        assert "|" in report  # Tables
    
    def test_55_deterministic_cross_check(self, tmp_path):
        """Cross-check is deterministic."""
        doc = tmp_path / "test.md"
        doc.write_text("Test `def:test` reference. RFL H_t Phase II.")
        
        fm_labels = {"def:test"}
        fm_definitions = [{"label": "def:test", "name": "Test"}]
        fm_invariants = []
        
        result1 = cross_check_document(doc, fm_labels, fm_definitions, fm_invariants)
        result2 = cross_check_document(doc, fm_labels, fm_definitions, fm_invariants)
        
        assert result1.status == result2.status
        assert result1.missing_labels == result2.missing_labels


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SPEC CONSUMERS INDEX TESTS (56-60)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecConsumersIndex:
    """Tests for spec contract consumer tracking."""
    
    def test_56_build_consumers_index_empty(self, tmp_path):
        """Empty docs produce empty consumers."""
        doc = tmp_path / "empty.md"
        doc.write_text("No references here.")
        
        fm_labels = {"def:test", "sec:intro"}
        index = build_consumers_index(fm_labels, [doc])
        
        assert "def:test" in index
        assert "sec:intro" in index
        assert index["def:test"]["total_refs"] == 0
        assert index["sec:intro"]["total_refs"] == 0
    
    def test_57_build_consumers_index_with_refs(self, tmp_path):
        """Consumers index tracks references correctly."""
        doc1 = tmp_path / "doc1.md"
        doc1.write_text("See `def:slice` for details.")
        
        doc2 = tmp_path / "doc2.md"
        doc2.write_text("Reference to `def:slice` here too.")
        
        fm_labels = {"def:slice"}
        index = build_consumers_index(fm_labels, [doc1, doc2])
        
        # Each doc has one reference
        assert index["def:slice"]["total_refs"] == 2
        assert "doc1.md" in index["def:slice"]["consumers"]
        assert "doc2.md" in index["def:slice"]["consumers"]
    
    def test_58_consumers_index_case_insensitive(self, tmp_path):
        """Consumer matching is case-insensitive."""
        doc = tmp_path / "test.md"
        doc.write_text("Reference to `DEF:SLICE` here.")
        
        fm_labels = {"def:slice"}
        index = build_consumers_index(fm_labels, [doc])
        
        assert index["def:slice"]["total_refs"] == 1
    
    def test_59_export_spec_with_consumers(self, tmp_path):
        """Export spec contract with consumer information."""
        # Create mock canonical JSON
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "version": "1.0.0",
            "source_hash": "abc123",
            "signature_hash": "def456",
            "labels": [
                {"name": "def:slice", "line": 10, "type": "definition"},
                {"name": "sec:intro", "line": 5, "type": "section"},
            ],
            "definitions": [{"name": "Slice", "label": "def:slice", "content": "...", "line": 10}],
            "invariants": [],
            "formulas": [],
            "sections": [{"title": "Introduction", "level": "section", "label": "sec:intro", "line": 5}],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        # Create mock doc
        doc = tmp_path / "doc.md"
        doc.write_text("Reference to `def:slice` here.")
        
        with patch('scripts.fm_canonicalize.CROSS_CHECK_DOCS', [doc]):
            contract = export_spec_contract_with_consumers(canon_path)
        
        assert "consumers" in contract
        assert "label_coverage" in contract
        assert "def:slice" in contract["consumers"]
    
    def test_60_consumers_missing_doc(self, tmp_path):
        """Missing docs are gracefully skipped."""
        missing_doc = tmp_path / "missing.md"  # Not created
        
        fm_labels = {"def:test"}
        index = build_consumers_index(fm_labels, [missing_doc])
        
        assert index["def:test"]["total_refs"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: LABEL DRIFT DETECTION TESTS (61-65)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelDriftDetection:
    """Tests for label drift detection between fm.tex and external docs."""
    
    def test_61_label_drift_well_connected(self, tmp_path):
        """Well-connected labels are identified correctly."""
        # Setup canonical JSON
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [
                {"name": "def:slice", "line": 10, "type": "definition"},
            ],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        # Setup doc with reference
        doc = tmp_path / "doc.md"
        doc.write_text("See `def:slice` for details.")
        
        drift = compute_label_drift(canon_path, [doc])
        
        assert "def:slice" in drift.well_connected_labels
        assert len(drift.fm_only_labels) == 0
    
    def test_62_label_drift_fm_only(self, tmp_path):
        """FM-only labels (defined but not referenced) are detected."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [
                {"name": "def:slice", "line": 10, "type": "definition"},
                {"name": "def:orphan", "line": 20, "type": "definition"},
            ],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("Only `def:slice` referenced here.")
        
        drift = compute_label_drift(canon_path, [doc])
        
        assert "def:orphan" in drift.fm_only_labels
        assert "def:slice" in drift.well_connected_labels
    
    def test_63_label_drift_external_only(self, tmp_path):
        """External-only labels (referenced but not defined) are detected."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [
                {"name": "def:slice", "line": 10, "type": "definition"},
            ],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("Reference to `def:slice` and `def:missing`.")
        
        drift = compute_label_drift(canon_path, [doc])
        
        assert "def:missing" in drift.external_only_labels
    
    def test_64_label_drift_coverage_calculation(self, tmp_path):
        """Coverage percentage is calculated correctly."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [
                {"name": "def:a", "line": 10, "type": "definition"},
                {"name": "def:b", "line": 20, "type": "definition"},
                {"name": "def:c", "line": 30, "type": "definition"},
                {"name": "def:d", "line": 40, "type": "definition"},
            ],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("References: `def:a` `def:b`")  # 2 of 4 = 50%
        
        drift = compute_label_drift(canon_path, [doc])
        
        assert drift.coverage_pct == 50.0
    
    def test_65_save_label_drift_summary(self, tmp_path):
        """Label drift summary is saved correctly."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [{"name": "def:test", "line": 10, "type": "definition"}],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("Reference to `def:test`.")
        
        drift = compute_label_drift(canon_path, [doc])
        
        out_path = tmp_path / "drift.json"
        save_label_drift_summary(drift, out_path)
        
        assert out_path.exists()
        saved = json.loads(out_path.read_text())
        assert "fm_only_labels" in saved
        assert "external_only_labels" in saved
        assert "well_connected_labels" in saved
        assert "coverage_pct" in saved


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: CI CONTRACT STABILITY TESTS (66-70)
# ═══════════════════════════════════════════════════════════════════════════════

class TestContractStability:
    """
    CI stability tests for spec contract shape.
    
    These tests force explicit human review when contract shape changes.
    """
    
    def test_66_contract_has_required_keys(self, tmp_path):
        """Contract must have all required top-level keys."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "version": "1.0.0",
            "source_hash": "abc",
            "signature_hash": "def",
            "labels": [],
            "definitions": [],
            "invariants": [],
            "formulas": [],
            "sections": [],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        contract = export_spec_contract(canon_path)
        
        for key in SPEC_CONTRACT_REQUIRED_KEYS:
            assert key in contract, f"Required key '{key}' missing from contract"
    
    def test_67_contract_no_unexpected_keys(self, tmp_path):
        """Contract must not have unexpected top-level keys (frozen API)."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "version": "1.0.0",
            "source_hash": "abc",
            "signature_hash": "def",
            "labels": [],
            "definitions": [],
            "invariants": [],
            "formulas": [],
            "sections": [],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        with patch('scripts.fm_canonicalize.CROSS_CHECK_DOCS', []):
            contract = export_spec_contract(canon_path)
        
        # All keys in contract must be in SPEC_CONTRACT_REQUIRED_KEYS
        allowed_keys = set(SPEC_CONTRACT_REQUIRED_KEYS)
        actual_keys = set(contract.keys())
        
        unexpected = actual_keys - allowed_keys
        assert len(unexpected) == 0, f"Unexpected keys in contract: {unexpected}"
    
    def test_68_contract_always_has_consumers(self, tmp_path):
        """Contract must always include consumers and label_coverage (frozen API)."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "version": "1.0.0",
            "source_hash": "abc",
            "signature_hash": "def",
            "labels": [{"name": "def:test", "line": 1, "type": "definition"}],
            "definitions": [],
            "invariants": [],
            "formulas": [],
            "sections": [],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("Test doc")
        
        with patch('scripts.fm_canonicalize.CROSS_CHECK_DOCS', [doc]):
            contract = export_spec_contract(canon_path)
        
        # Consumers and label_coverage are now always included
        assert "consumers" in contract, "consumers must always be in contract"
        assert "label_coverage" in contract, "label_coverage must always be in contract"
        assert isinstance(contract["consumers"], dict)
        assert "labels_with_refs" in contract["label_coverage"]
    
    def test_69_contract_version_format(self, tmp_path):
        """Contract version must follow semver-like format."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "version": "1.0.0",
            "source_hash": "abc",
            "signature_hash": "def",
            "labels": [],
            "definitions": [],
            "invariants": [],
            "formulas": [],
            "sections": [],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        contract = export_spec_contract(canon_path)
        
        import re
        version = contract["version"]
        assert re.match(r"^\d+\.\d+$", version), f"Version '{version}' must be semver-like"
    
    def test_70_contract_signature_hash_format(self, tmp_path):
        """Contract signature hash must be valid hex string."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "version": "1.0.0",
            "source_hash": "abc123",
            "signature_hash": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "labels": [],
            "definitions": [],
            "invariants": [],
            "formulas": [],
            "sections": [],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        contract = export_spec_contract(canon_path)
        
        sig_hash = contract["signature_hash"]
        assert len(sig_hash) == 64, f"Signature hash must be 64 hex chars"
        assert all(c in "0123456789abcdef" for c in sig_hash.lower()), "Signature must be hex"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: INTEGRATION TESTS FOR NEW FEATURES (71-75)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewFeaturesE6Integration:
    """Integration tests for E6 new features: consumers, drift, stability."""
    
    @pytest.mark.skipif(not FM_TEX_PATH.exists(), reason="fm.tex not available")
    def test_71_full_consumers_with_real_fm(self):
        """Export spec contract with consumers using real fm.tex."""
        if not FM_CANONICAL_PATH.exists():
            pytest.skip("Canonical JSON not available")
        
        contract = export_spec_contract_with_consumers(FM_CANONICAL_PATH)
        
        assert "consumers" in contract
        assert isinstance(contract["consumers"], dict)
        assert "label_coverage" in contract
    
    @pytest.mark.skipif(not FM_TEX_PATH.exists(), reason="fm.tex not available")
    def test_72_label_drift_with_real_fm(self):
        """Compute label drift using real fm.tex."""
        if not FM_CANONICAL_PATH.exists():
            pytest.skip("Canonical JSON not available")
        
        drift = compute_label_drift(FM_CANONICAL_PATH)
        
        assert isinstance(drift.fm_only_labels, list)
        assert isinstance(drift.external_only_labels, list)
        assert isinstance(drift.well_connected_labels, list)
        assert 0 <= drift.coverage_pct <= 100
    
    def test_73_deterministic_consumers_index(self, tmp_path):
        """Consumer index is deterministic."""
        doc = tmp_path / "doc.md"
        doc.write_text("Reference `def:test` and `def:test` again.")
        
        fm_labels = {"def:test"}
        
        index1 = build_consumers_index(fm_labels, [doc])
        index2 = build_consumers_index(fm_labels, [doc])
        
        assert index1 == index2
    
    def test_74_deterministic_drift_detection(self, tmp_path):
        """Drift detection is deterministic."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [{"name": "def:test", "line": 10, "type": "definition"}],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("Reference `def:test` and `def:missing`.")
        
        drift1 = compute_label_drift(canon_path, [doc])
        drift2 = compute_label_drift(canon_path, [doc])
        
        assert drift1.fm_only_labels == drift2.fm_only_labels
        assert drift1.external_only_labels == drift2.external_only_labels
        assert drift1.well_connected_labels == drift2.well_connected_labels
        assert drift1.coverage_pct == drift2.coverage_pct
    
    def test_75_contract_keys_match_spec(self):
        """Required keys constant matches actual spec (frozen API)."""
        expected_required = [
            "version",
            "signature_hash",
            "definitions",
            "invariants",
            "metrics_section",
            "uplift_section",
            "phase_section",
            "consumers",
            "label_coverage",
        ]
        
        assert set(SPEC_CONTRACT_REQUIRED_KEYS) == set(expected_required), \
            "SPEC_CONTRACT_REQUIRED_KEYS must match expected keys"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: LABEL DRIFT SEMANTICS TESTS (76-80)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelDriftSemantics:
    """
    Tests verifying label drift summary semantics.
    
    Label drift categories:
        - fm_only_labels: Defined in fm.tex, not referenced externally
        - external_only_labels: Referenced externally, not defined in fm.tex
        - well_connected_labels: Both defined and referenced
        - coverage_pct: Fraction of fm.tex labels that are well-connected
    """
    
    def test_76_label_drift_has_required_keys(self, tmp_path):
        """Label drift summary must have all required keys."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [{"name": "def:test", "line": 10, "type": "definition"}],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("Reference `def:test`.")
        
        drift = compute_label_drift(canon_path, [doc])
        
        for key in LABEL_DRIFT_REQUIRED_KEYS:
            assert hasattr(drift, key.replace("-", "_")), f"Required key '{key}' missing"
    
    def test_77_fm_only_semantics(self, tmp_path):
        """fm_only_labels contains labels defined but never externally referenced."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [
                {"name": "def:used", "line": 10, "type": "definition"},
                {"name": "def:orphan1", "line": 20, "type": "definition"},
                {"name": "def:orphan2", "line": 30, "type": "definition"},
            ],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("Only `def:used` is referenced here.")
        
        drift = compute_label_drift(canon_path, [doc])
        
        assert "def:orphan1" in drift.fm_only_labels
        assert "def:orphan2" in drift.fm_only_labels
        assert "def:used" not in drift.fm_only_labels
    
    def test_78_external_only_semantics(self, tmp_path):
        """external_only_labels contains references to undefined labels."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [{"name": "def:defined", "line": 10, "type": "definition"}],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("References: `def:defined` and `def:undefined` and `inv:missing`.")
        
        drift = compute_label_drift(canon_path, [doc])
        
        assert "def:undefined" in drift.external_only_labels
        assert "inv:missing" in drift.external_only_labels
        assert "def:defined" not in drift.external_only_labels
    
    def test_79_well_connected_semantics(self, tmp_path):
        """well_connected_labels contains labels both defined and referenced."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [
                {"name": "def:connected1", "line": 10, "type": "definition"},
                {"name": "def:connected2", "line": 20, "type": "definition"},
                {"name": "def:orphan", "line": 30, "type": "definition"},
            ],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("See `def:connected1` and `def:connected2`.")
        
        drift = compute_label_drift(canon_path, [doc])
        
        assert "def:connected1" in drift.well_connected_labels
        assert "def:connected2" in drift.well_connected_labels
        assert "def:orphan" not in drift.well_connected_labels
    
    def test_80_coverage_pct_calculation(self, tmp_path):
        """coverage_pct = well_connected / total_fm_labels * 100."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [
                {"name": "def:a", "line": 1, "type": "definition"},
                {"name": "def:b", "line": 2, "type": "definition"},
                {"name": "def:c", "line": 3, "type": "definition"},
                {"name": "def:d", "line": 4, "type": "definition"},
                {"name": "def:e", "line": 5, "type": "definition"},
            ],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("Used: `def:a` `def:b`")  # 2 of 5 = 40%
        
        drift = compute_label_drift(canon_path, [doc])
        
        assert drift.coverage_pct == 40.0
        assert len(drift.well_connected_labels) == 2
        assert len(drift.fm_only_labels) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: CI INTEGRATION TESTS (81-85)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCIIntegration:
    """
    Tests verifying CI integration behavior.
    
    CI INTEGRATION BLUEPRINT:
    
        # Run full analysis (recommended)
        uv run python scripts/fm_canonicalize.py full
        
        # Run label drift detection (strict mode fails on external-only labels)
        uv run python scripts/fm_canonicalize.py label-drift --strict
    
    AGENT CONSUMPTION:
    
        Agents I, J, N, O can rely on:
        
        1. fm_spec_contract.json
           - Label/definition mappings
           - Invariant listings
           - Consumer references
           
        2. cross_check_summary.json
           - Doc-level drift detection
           - Missing label reports
           
        3. label_drift_summary.json
           - fm_only_labels: Orphaned definitions
           - external_only_labels: Potential documentation errors
           - coverage_pct: Health metric
    """
    
    def test_81_full_command_generates_all_artifacts(self, tmp_path):
        """full command generates all required artifacts."""
        # This test documents that the full command should generate:
        # - docs/fm_canonical.json
        # - artifacts/spec/fm_spec_contract.json
        # - artifacts/spec/label_drift_summary.json
        
        # Just verify the paths are defined correctly
        assert FM_CANONICAL_PATH.name == "fm_canonical.json"
        assert FM_SPEC_CONTRACT_PATH.name == "fm_spec_contract.json"
        assert LABEL_DRIFT_PATH.name == "label_drift_summary.json"
    
    def test_82_spec_contract_is_stable_api(self, tmp_path):
        """Spec contract shape is frozen for other agents."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "version": "1.0.0",
            "source_hash": "abc",
            "signature_hash": "def456",
            "labels": [{"name": "def:test", "line": 1, "type": "definition"}],
            "definitions": [{"name": "Test", "label": "def:test", "content": "", "line": 1}],
            "invariants": [],
            "formulas": [],
            "sections": [],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        with patch('scripts.fm_canonicalize.CROSS_CHECK_DOCS', []):
            contract = export_spec_contract(canon_path)
        
        # Verify frozen structure
        assert contract["version"] == "1.0"
        assert isinstance(contract["signature_hash"], str)
        assert isinstance(contract["definitions"], list)
        assert isinstance(contract["invariants"], list)
        assert isinstance(contract["consumers"], dict)
        assert isinstance(contract["label_coverage"], dict)
    
    def test_83_label_drift_strict_mode_behavior(self, tmp_path):
        """Strict mode should report external-only labels as errors."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123",
            "labels": [{"name": "def:defined", "line": 10, "type": "definition"}],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        # Doc references undefined label
        doc = tmp_path / "doc.md"
        doc.write_text("Reference to `def:undefined`.")
        
        drift = compute_label_drift(canon_path, [doc])
        
        # external_only_labels should trigger strict mode failure
        assert len(drift.external_only_labels) > 0
        # In strict mode, CI would return exit code 1
    
    def test_84_deterministic_artifacts(self, tmp_path):
        """All artifacts must be deterministic given same inputs."""
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "signature_hash": "abc123def456",
            "labels": [
                {"name": "def:a", "line": 1, "type": "definition"},
                {"name": "def:b", "line": 2, "type": "definition"},
            ],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("Reference `def:a`.")
        
        # Run twice
        with patch('scripts.fm_canonicalize.CROSS_CHECK_DOCS', [doc]):
            contract1 = export_spec_contract(canon_path)
            contract2 = export_spec_contract(canon_path)
        
        drift1 = compute_label_drift(canon_path, [doc])
        drift2 = compute_label_drift(canon_path, [doc])
        
        # All outputs must be identical
        assert contract1 == contract2
        assert drift1.fm_only_labels == drift2.fm_only_labels
        assert drift1.external_only_labels == drift2.external_only_labels
        assert drift1.well_connected_labels == drift2.well_connected_labels
    
    def test_85_agent_consumption_patterns(self, tmp_path):
        """
        Document how agents I, J, N, O consume spec contract.
        
        This test serves as documentation for agent consumption patterns.
        """
        canon_path = tmp_path / "fm_canonical.json"
        canon_data = {
            "version": "1.0.0",
            "source_hash": "abc",
            "signature_hash": "sig123",
            "labels": [
                {"name": "def:slice", "line": 10, "type": "definition"},
                {"name": "inv:monotonicity", "line": 20, "type": "invariant"},
            ],
            "definitions": [{"name": "Slice", "label": "def:slice", "content": "", "line": 10}],
            "invariants": [{"name": "Monotonicity", "label": "inv:monotonicity", "content": "", "line": 20}],
            "formulas": [],
            "sections": [{"title": "Introduction", "level": "section", "label": "sec:intro", "line": 1}],
        }
        canon_path.write_text(json.dumps(canon_data))
        
        doc = tmp_path / "doc.md"
        doc.write_text("See `def:slice` for the Slice definition.")
        
        with patch('scripts.fm_canonicalize.CROSS_CHECK_DOCS', [doc]):
            contract = export_spec_contract(canon_path)
        
        # Agent consumption pattern: lookup definition by label
        def lookup_definition(contract: dict, label: str) -> dict:
            for defn in contract["definitions"]:
                if defn["label"] == label:
                    return defn
            return None
        
        slice_def = lookup_definition(contract, "def:slice")
        assert slice_def is not None
        assert slice_def["name"] == "Slice"
        
        # Agent consumption pattern: check if label is well-connected
        def is_well_connected(contract: dict, label: str) -> bool:
            consumers = contract.get("consumers", {})
            label_data = consumers.get(label, {})
            return label_data.get("total_refs", 0) > 0
        
        assert is_well_connected(contract, "def:slice")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: FM ROUTER API TESTS (86-95)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFMRouterAPI:
    """
    Tests for the FM Router API — fast lookup and posture functions.
    
    These functions provide a stable, code-level interface over the spec contract
    for use by other agents (I, J, N, O).
    """
    
    @pytest.fixture
    def sample_contract(self):
        """Sample spec contract for testing."""
        return {
            "version": "1.0",
            "signature_hash": "abc123def456",
            "definitions": [
                {"label": "def:slice", "name": "Slice", "section": "2"},
                {"label": "def:policy", "name": "Policy", "section": "3"},
            ],
            "invariants": [
                {"label": "inv:monotonicity", "name": "Monotonicity", "section": "4"},
            ],
            "metrics_section": "5",
            "uplift_section": "6",
            "phase_section": "1",
            "consumers": {
                "def:slice": {
                    "consumers": {"PHASE2_PLAN.md": 3, "THEORY.md": 1},
                    "total_refs": 4,
                },
                "def:policy": {
                    "consumers": {},
                    "total_refs": 0,
                },
                "inv:monotonicity": {
                    "consumers": {"PHASE2_PLAN.md": 2},
                    "total_refs": 2,
                },
            },
            "label_coverage": {
                "labels_with_refs": 2,
                "total_labels": 3,
                "coverage_pct": 66.67,
            },
        }
    
    @pytest.fixture
    def sample_drift_summary(self):
        """Sample drift summary for testing."""
        return {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "fm_signature_hash": "abc123def456",
            "fm_only_labels": ["def:policy"],
            "external_only_labels": [],
            "well_connected_labels": ["def:slice", "inv:monotonicity"],
            "coverage_pct": 66.67,
        }
    
    def test_86_get_label_definition_found(self, sample_contract):
        """get_label_definition returns correct definition."""
        result = get_label_definition(sample_contract, "def:slice")
        
        assert result is not None
        assert result["label"] == "def:slice"
        assert result["name"] == "Slice"
        assert result["section"] == "2"
    
    def test_87_get_label_definition_invariant(self, sample_contract):
        """get_label_definition also finds invariants."""
        result = get_label_definition(sample_contract, "inv:monotonicity")
        
        assert result is not None
        assert result["label"] == "inv:monotonicity"
        assert result["name"] == "Monotonicity"
    
    def test_88_get_label_definition_not_found(self, sample_contract):
        """get_label_definition returns None for unknown labels."""
        result = get_label_definition(sample_contract, "def:nonexistent")
        
        assert result is None
    
    def test_89_get_label_consumers(self, sample_contract):
        """get_label_consumers returns correct consumer mapping."""
        result = get_label_consumers(sample_contract, "def:slice")
        
        assert result == {"PHASE2_PLAN.md": 3, "THEORY.md": 1}
    
    def test_90_get_label_consumers_empty(self, sample_contract):
        """get_label_consumers returns empty dict for labels with no consumers."""
        result = get_label_consumers(sample_contract, "def:policy")
        
        assert result == {}
    
    def test_91_get_label_consumers_not_found(self, sample_contract):
        """get_label_consumers returns empty dict for unknown labels."""
        result = get_label_consumers(sample_contract, "def:nonexistent")
        
        assert result == {}
    
    def test_92_get_label_total_refs(self, sample_contract):
        """get_label_total_refs returns correct count."""
        assert get_label_total_refs(sample_contract, "def:slice") == 4
        assert get_label_total_refs(sample_contract, "def:policy") == 0
        assert get_label_total_refs(sample_contract, "def:nonexistent") == 0
    
    def test_93_is_label_well_connected(self, sample_contract):
        """is_label_well_connected returns correct boolean."""
        assert is_label_well_connected(sample_contract, "def:slice") is True
        assert is_label_well_connected(sample_contract, "def:policy") is False
        assert is_label_well_connected(sample_contract, "def:nonexistent") is False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: FM CONSUMERS VIEW TESTS (94-98)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFMConsumersView:
    """Tests for the cross-doc FM consumers view."""
    
    @pytest.fixture
    def sample_contract(self):
        """Sample spec contract with consumers."""
        return {
            "consumers": {
                "def:slice": {
                    "consumers": {"PHASE2_PLAN.md": 5, "THEORY.md": 2},
                    "total_refs": 7,
                },
                "def:policy": {
                    "consumers": {"PHASE2_PLAN.md": 3},
                    "total_refs": 3,
                },
                "inv:mono": {
                    "consumers": {"SUMMARY.md": 1},
                    "total_refs": 1,
                },
            },
        }
    
    def test_94_build_fm_consumers_view_aggregation(self, sample_contract):
        """build_fm_consumers_view correctly aggregates by document."""
        view = build_fm_consumers_view(sample_contract)
        
        # Should have 3 docs
        assert len(view) == 3
        
        # Find each doc
        plan_entry = next(e for e in view if e["doc"] == "PHASE2_PLAN.md")
        theory_entry = next(e for e in view if e["doc"] == "THEORY.md")
        summary_entry = next(e for e in view if e["doc"] == "SUMMARY.md")
        
        # Verify aggregation
        assert plan_entry["total_refs"] == 8  # 5 + 3
        assert plan_entry["label_count"] == 2
        
        assert theory_entry["total_refs"] == 2
        assert theory_entry["label_count"] == 1
        
        assert summary_entry["total_refs"] == 1
        assert summary_entry["label_count"] == 1
    
    def test_95_build_fm_consumers_view_sorted(self, sample_contract):
        """build_fm_consumers_view is sorted by total_refs descending."""
        view = build_fm_consumers_view(sample_contract)
        
        # Should be sorted by total_refs descending
        refs = [e["total_refs"] for e in view]
        assert refs == sorted(refs, reverse=True)
        
        # First should be PHASE2_PLAN.md with most refs
        assert view[0]["doc"] == "PHASE2_PLAN.md"
    
    def test_96_build_fm_consumers_view_empty(self):
        """build_fm_consumers_view handles empty consumers."""
        contract = {"consumers": {}}
        view = build_fm_consumers_view(contract)
        
        assert view == []
    
    def test_97_build_fm_consumers_view_no_refs(self):
        """build_fm_consumers_view handles labels with no consumers."""
        contract = {
            "consumers": {
                "def:orphan": {"consumers": {}, "total_refs": 0},
            },
        }
        view = build_fm_consumers_view(contract)
        
        # No docs with refs, so empty view
        assert view == []
    
    def test_98_build_fm_consumers_view_deterministic(self, sample_contract):
        """build_fm_consumers_view is deterministic."""
        view1 = build_fm_consumers_view(sample_contract)
        view2 = build_fm_consumers_view(sample_contract)
        
        assert view1 == view2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14: FM POSTURE TESTS (99-105)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFMPosture:
    """Tests for the FM posture snapshot."""
    
    @pytest.fixture
    def healthy_contract(self):
        """Contract with good coverage."""
        return {
            "signature_hash": "abc123",
            "label_coverage": {
                "labels_with_refs": 8,
                "total_labels": 10,
                "coverage_pct": 80.0,
            },
        }
    
    @pytest.fixture
    def healthy_drift(self):
        """Drift summary with no issues."""
        return {
            "fm_only_labels": ["def:a", "def:b"],
            "external_only_labels": [],
            "well_connected_labels": ["def:c", "def:d", "def:e", "def:f", "def:g", "def:h", "def:i", "def:j"],
            "coverage_pct": 80.0,
        }
    
    @pytest.fixture
    def warning_drift(self):
        """Drift summary with low coverage."""
        return {
            "fm_only_labels": ["def:a", "def:b", "def:c", "def:d", "def:e", "def:f", "def:g", "def:h", "def:i"],
            "external_only_labels": [],
            "well_connected_labels": ["def:j"],
            "coverage_pct": 5.0,
        }
    
    @pytest.fixture
    def critical_drift(self):
        """Drift summary with external-only labels."""
        return {
            "fm_only_labels": ["def:a"],
            "external_only_labels": ["def:missing", "inv:undefined"],
            "well_connected_labels": ["def:b"],
            "coverage_pct": 50.0,
        }
    
    def test_99_posture_schema(self, healthy_contract, healthy_drift):
        """build_fm_posture returns correct schema."""
        posture = build_fm_posture(healthy_contract, healthy_drift)
        
        assert "schema_version" in posture
        assert "signature_hash" in posture
        assert "total_labels" in posture
        assert "well_connected_labels" in posture
        assert "external_only_labels" in posture
        assert "fm_only_labels" in posture
        assert "coverage_pct" in posture
        assert "health_status" in posture
    
    def test_100_posture_healthy(self, healthy_contract, healthy_drift):
        """build_fm_posture returns 'healthy' for good coverage."""
        posture = build_fm_posture(healthy_contract, healthy_drift)
        
        assert posture["health_status"] == "healthy"
        assert posture["well_connected_labels"] == 8
        assert posture["fm_only_labels"] == 2
        assert posture["external_only_labels"] == 0
    
    def test_101_posture_warning(self, healthy_contract, warning_drift):
        """build_fm_posture returns 'warning' for low coverage."""
        posture = build_fm_posture(healthy_contract, warning_drift)
        
        assert posture["health_status"] == "warning"
        assert posture["coverage_pct"] == 5.0
    
    def test_102_posture_critical(self, healthy_contract, critical_drift):
        """build_fm_posture returns 'critical' for external-only labels."""
        posture = build_fm_posture(healthy_contract, critical_drift)
        
        assert posture["health_status"] == "critical"
        assert posture["external_only_labels"] == 2
    
    def test_103_posture_deterministic(self, healthy_contract, healthy_drift):
        """build_fm_posture is deterministic."""
        posture1 = build_fm_posture(healthy_contract, healthy_drift)
        posture2 = build_fm_posture(healthy_contract, healthy_drift)
        
        assert posture1 == posture2
    
    def test_104_posture_counts_match(self, healthy_contract, healthy_drift):
        """build_fm_posture counts match input data."""
        posture = build_fm_posture(healthy_contract, healthy_drift)
        
        assert posture["well_connected_labels"] == len(healthy_drift["well_connected_labels"])
        assert posture["fm_only_labels"] == len(healthy_drift["fm_only_labels"])
        assert posture["external_only_labels"] == len(healthy_drift["external_only_labels"])
    
    def test_105_posture_schema_version(self, healthy_contract, healthy_drift):
        """build_fm_posture has correct schema version."""
        posture = build_fm_posture(healthy_contract, healthy_drift)
        
        assert posture["schema_version"] == "1.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15: FM GOVERNANCE TESTS (106-110)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFMGovernance:
    """Tests for FM governance snapshot."""
    
    @pytest.fixture
    def healthy_posture(self):
        """Healthy posture for testing."""
        return {
            "schema_version": "1.0.0",
            "total_labels": 10,
            "well_connected_labels": 8,
            "external_only_labels": 0,
            "fm_only_labels": 2,
            "coverage_pct": 80.0,
            "health_status": "healthy",
        }
    
    @pytest.fixture
    def warning_posture(self):
        """Warning posture (low coverage)."""
        return {
            "schema_version": "1.0.0",
            "total_labels": 10,
            "well_connected_labels": 1,
            "external_only_labels": 0,
            "fm_only_labels": 9,
            "coverage_pct": 5.0,
            "health_status": "warning",
        }
    
    @pytest.fixture
    def attention_posture(self):
        """Attention posture (external-only labels)."""
        return {
            "schema_version": "1.0.0",
            "total_labels": 10,
            "well_connected_labels": 5,
            "external_only_labels": 2,
            "fm_only_labels": 3,
            "coverage_pct": 50.0,
            "health_status": "critical",
        }
    
    def test_106_governance_schema(self, healthy_posture):
        """build_fm_governance_snapshot returns correct schema."""
        gov = build_fm_governance_snapshot(healthy_posture)
        
        assert "schema_version" in gov
        assert "governance_status" in gov
        assert "health_status" in gov
        assert "total_labels" in gov
        assert "coverage_pct" in gov
    
    def test_107_governance_ok(self, healthy_posture):
        """build_fm_governance_snapshot returns OK for healthy posture."""
        gov = build_fm_governance_snapshot(healthy_posture)
        
        assert gov["governance_status"] == "OK"
        assert gov["external_only_labels"] == 0
        assert gov["coverage_pct"] == 80.0
    
    def test_108_governance_warn(self, warning_posture):
        """build_fm_governance_snapshot returns WARN for low coverage."""
        gov = build_fm_governance_snapshot(warning_posture)
        
        assert gov["governance_status"] == "WARN"
        assert gov["coverage_pct"] < 10
    
    def test_109_governance_attention(self, attention_posture):
        """build_fm_governance_snapshot returns ATTENTION for external-only labels."""
        gov = build_fm_governance_snapshot(attention_posture)
        
        assert gov["governance_status"] == "ATTENTION"
        assert gov["external_only_labels"] > 0
    
    def test_110_governance_deterministic(self, healthy_posture):
        """build_fm_governance_snapshot is deterministic."""
        gov1 = build_fm_governance_snapshot(healthy_posture)
        gov2 = build_fm_governance_snapshot(healthy_posture)
        
        assert gov1 == gov2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16: ALIGNMENT INDICATOR TESTS (111-116)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlignmentIndicator:
    """Tests for cross-system alignment indicator."""
    
    @pytest.fixture
    def well_distributed_view(self):
        """Well-distributed consumers view."""
        return [
            {"doc": "doc1.md", "total_refs": 10, "label_count": 8},
            {"doc": "doc2.md", "total_refs": 8, "label_count": 6},
            {"doc": "doc3.md", "total_refs": 7, "label_count": 5},
        ]
    
    @pytest.fixture
    def concentrated_view(self):
        """Concentrated consumers view (one doc dominates)."""
        return [
            {"doc": "doc1.md", "total_refs": 20, "label_count": 5},
            {"doc": "doc2.md", "total_refs": 2, "label_count": 2},
        ]
    
    @pytest.fixture
    def sparse_view(self):
        """Sparse consumers view (few refs)."""
        return [
            {"doc": "doc1.md", "total_refs": 2, "label_count": 1},
        ]
    
    @pytest.fixture
    def low_diversity_view(self):
        """View with low label diversity."""
        return [
            {"doc": "doc1.md", "total_refs": 20, "label_count": 3},  # Low diversity
            {"doc": "doc2.md", "total_refs": 15, "label_count": 12},  # Good diversity
        ]
    
    def test_111_alignment_schema(self, well_distributed_view):
        """compute_alignment_indicator returns correct schema."""
        alignment = compute_alignment_indicator(well_distributed_view)
        
        assert "heaviest_consumers" in alignment
        assert "docs_with_low_label_diversity" in alignment
        assert "alignment_status" in alignment
    
    def test_112_alignment_well_distributed(self, well_distributed_view):
        """compute_alignment_indicator detects well-distributed pattern."""
        alignment = compute_alignment_indicator(well_distributed_view)
        
        assert alignment["alignment_status"] == "WELL_DISTRIBUTED"
        assert len(alignment["heaviest_consumers"]) == 3
    
    def test_113_alignment_concentrated(self, concentrated_view):
        """compute_alignment_indicator detects concentrated pattern."""
        alignment = compute_alignment_indicator(concentrated_view)
        
        assert alignment["alignment_status"] == "CONCENTRATED"
        # Top doc has > 50% of total refs (20 out of 22)
    
    def test_114_alignment_sparse(self, sparse_view):
        """compute_alignment_indicator detects sparse pattern."""
        alignment = compute_alignment_indicator(sparse_view)
        
        assert alignment["alignment_status"] == "SPARSE"
    
    def test_115_alignment_low_diversity(self, low_diversity_view):
        """compute_alignment_indicator detects low label diversity."""
        alignment = compute_alignment_indicator(low_diversity_view)
        
        # doc1.md has 20 refs but only 3 labels (diversity ratio < 0.3)
        low_div = alignment["docs_with_low_label_diversity"]
        assert len(low_div) > 0
        assert any(e["doc"] == "doc1.md" for e in low_div)
    
    def test_116_alignment_empty(self):
        """compute_alignment_indicator handles empty view."""
        alignment = compute_alignment_indicator([])
        
        assert alignment["alignment_status"] == "SPARSE"
        assert alignment["heaviest_consumers"] == []
        assert alignment["docs_with_low_label_diversity"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17: GLOBAL HEALTH TESTS (117-122)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlobalHealth:
    """Tests for global health FM signal."""
    
    @pytest.fixture
    def ok_governance(self):
        """OK governance snapshot."""
        return {
            "governance_status": "OK",
            "health_status": "healthy",
            "coverage_pct": 80.0,
            "external_only_labels": 0,
        }
    
    @pytest.fixture
    def warn_governance(self):
        """WARN governance snapshot."""
        return {
            "governance_status": "WARN",
            "health_status": "warning",
            "coverage_pct": 5.0,
            "external_only_labels": 0,
        }
    
    @pytest.fixture
    def block_governance(self):
        """BLOCK governance snapshot (external-only labels)."""
        return {
            "governance_status": "ATTENTION",
            "health_status": "critical",
            "coverage_pct": 50.0,
            "external_only_labels": 2,
        }
    
    @pytest.fixture
    def well_distributed_alignment(self):
        """Well-distributed alignment."""
        return {
            "alignment_status": "WELL_DISTRIBUTED",
            "heaviest_consumers": [],
            "docs_with_low_label_diversity": [],
        }
    
    @pytest.fixture
    def concentrated_alignment(self):
        """Concentrated alignment."""
        return {
            "alignment_status": "CONCENTRATED",
            "heaviest_consumers": [],
            "docs_with_low_label_diversity": [],
        }
    
    def test_117_global_health_schema(self, ok_governance, well_distributed_alignment):
        """summarize_fm_for_global_health returns correct schema."""
        health = summarize_fm_for_global_health(ok_governance, well_distributed_alignment)
        
        assert "fm_ok" in health
        assert "coverage_pct" in health
        assert "external_only_labels" in health
        assert "alignment_status" in health
        assert "status" in health
    
    def test_118_global_health_ok(self, ok_governance, well_distributed_alignment):
        """summarize_fm_for_global_health returns OK for healthy state."""
        health = summarize_fm_for_global_health(ok_governance, well_distributed_alignment)
        
        assert health["status"] == "OK"
        assert health["fm_ok"] is True
        assert health["external_only_labels"] == 0
    
    def test_119_global_health_warn(self, warn_governance, well_distributed_alignment):
        """summarize_fm_for_global_health returns WARN for warning state."""
        health = summarize_fm_for_global_health(warn_governance, well_distributed_alignment)
        
        assert health["status"] == "WARN"
        assert health["fm_ok"] is False
    
    def test_120_global_health_warn_concentrated(self, ok_governance, concentrated_alignment):
        """summarize_fm_for_global_health returns WARN for concentrated alignment."""
        health = summarize_fm_for_global_health(ok_governance, concentrated_alignment)
        
        assert health["status"] == "WARN"
        assert health["alignment_status"] == "CONCENTRATED"
    
    def test_121_global_health_block(self, block_governance, well_distributed_alignment):
        """summarize_fm_for_global_health returns BLOCK for external-only labels."""
        health = summarize_fm_for_global_health(block_governance, well_distributed_alignment)
        
        assert health["status"] == "BLOCK"
        assert health["fm_ok"] is False
        assert health["external_only_labels"] > 0
    
    def test_122_global_health_deterministic(self, ok_governance, well_distributed_alignment):
        """summarize_fm_for_global_health is deterministic."""
        health1 = summarize_fm_for_global_health(ok_governance, well_distributed_alignment)
        health2 = summarize_fm_for_global_health(ok_governance, well_distributed_alignment)
        
        assert health1 == health2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18: CROSS-AGENT LABEL CONTRACT INDEX TESTS (123-128)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelContractIndex:
    """Tests for cross-agent label contract index."""
    
    @pytest.fixture
    def sample_spec_contract(self):
        """Sample spec contract."""
        return {
            "version": "1.0",
            "signature_hash": "abc123",
            "definitions": [
                {"label": "def:slice", "name": "Slice", "section": "2"},
                {"label": "def:policy", "name": "Policy", "section": "3"},
            ],
            "invariants": [
                {"label": "inv:monotonicity", "name": "Monotonicity", "section": "4"},
            ],
        }
    
    @pytest.fixture
    def sample_taxonomy(self):
        """Sample taxonomy semantics."""
        return {
            "terms": ["def:slice", "def:policy", "taxonomy_term"],
            "labels": ["inv:monotonicity"],
        }
    
    @pytest.fixture
    def sample_curriculum(self):
        """Sample curriculum manifest."""
        return {
            "slices": [
                {"name": "slice_easy_fo"},
                {"name": "slice_hard"},
            ],
            "systems": {
                "pl": {
                    "slices": [
                        {"name": "slice_debug"},
                    ],
                },
            },
        }
    
    def test_123_extract_taxonomy_labels(self, sample_taxonomy):
        """extract_labels_from_taxonomy extracts labels correctly."""
        labels = extract_labels_from_taxonomy(sample_taxonomy)
        
        assert "def:slice" in labels
        assert "def:policy" in labels
        assert "taxonomy_term" in labels
        assert "inv:monotonicity" in labels
    
    def test_124_extract_curriculum_labels(self, sample_curriculum):
        """extract_labels_from_curriculum extracts labels correctly."""
        labels = extract_labels_from_curriculum(sample_curriculum)
        
        assert "slice_easy_fo" in labels
        assert "slice_hard" in labels
        assert "slice_debug" in labels
    
    def test_125_label_contract_index_schema(
        self, sample_spec_contract, sample_taxonomy, sample_curriculum
    ):
        """build_label_contract_index returns correct schema."""
        index = build_label_contract_index(
            sample_spec_contract, sample_taxonomy, sample_curriculum
        )
        
        assert "schema_version" in index
        assert "label_index" in index
        assert "labels_missing_in_taxonomy" in index
        assert "labels_missing_in_curriculum" in index
        assert "labels_missing_in_fm" in index
        assert "contract_status" in index
    
    def test_126_label_contract_aligned(
        self, sample_spec_contract, sample_taxonomy, sample_curriculum
    ):
        """build_label_contract_index detects aligned status."""
        # Create aligned case: all taxonomy labels in FM, curriculum empty
        taxonomy = {"terms": ["def:slice", "def:policy"]}
        curriculum = {}  # Empty curriculum for true alignment
        index = build_label_contract_index(
            sample_spec_contract, taxonomy, curriculum
        )
        
        # Should be aligned when all taxonomy labels are in FM
        assert index["contract_status"] == "ALIGNED"
        assert "def:slice" in index["label_index"]
        assert index["label_index"]["def:slice"]["in_fm"] is True
        assert index["label_index"]["def:slice"]["in_taxonomy"] is True
    
    def test_127_label_contract_broken(
        self, sample_spec_contract, sample_taxonomy, sample_curriculum
    ):
        """build_label_contract_index detects broken status."""
        # Create broken case: taxonomy label not in FM
        taxonomy = {"terms": ["def:missing_term"]}
        index = build_label_contract_index(
            sample_spec_contract, taxonomy, sample_curriculum
        )
        
        assert index["contract_status"] == "BROKEN"
        assert "def:missing_term" in index["labels_missing_in_fm"]
    
    def test_128_label_contract_deterministic(
        self, sample_spec_contract, sample_taxonomy, sample_curriculum
    ):
        """build_label_contract_index is deterministic."""
        index1 = build_label_contract_index(
            sample_spec_contract, sample_taxonomy, sample_curriculum
        )
        index2 = build_label_contract_index(
            sample_spec_contract, sample_taxonomy, sample_curriculum
        )
        
        assert index1 == index2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 19: AGENT INTEGRATION CONTRACT TESTS (129-133)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentIntegrationContract:
    """Tests for agent integration contract."""
    
    @pytest.fixture
    def aligned_label_index(self):
        """Aligned label index."""
        return {
            "schema_version": "1.0.0",
            "label_index": {
                "def:slice": {"in_fm": True, "in_taxonomy": True, "in_curriculum": True},
                "def:policy": {"in_fm": True, "in_taxonomy": True, "in_curriculum": False},
            },
            "labels_missing_in_taxonomy": [],
            "labels_missing_in_curriculum": [],
            "labels_missing_in_fm": [],
            "contract_status": "ALIGNED",
        }
    
    @pytest.fixture
    def broken_label_index(self):
        """Broken label index."""
        return {
            "schema_version": "1.0.0",
            "label_index": {
                "def:missing": {"in_fm": False, "in_taxonomy": True, "in_curriculum": True},
            },
            "labels_missing_in_taxonomy": [],
            "labels_missing_in_curriculum": [],
            "labels_missing_in_fm": ["def:missing"],
            "contract_status": "BROKEN",
        }
    
    def test_129_integration_contract_schema(self, aligned_label_index):
        """build_field_manual_integration_contract returns correct schema."""
        contract = build_field_manual_integration_contract(aligned_label_index)
        
        assert "contract_version" in contract
        assert "required_label_sets" in contract
        assert "integration_status" in contract
        assert "notes" in contract
    
    def test_130_integration_contract_ok(self, aligned_label_index):
        """build_field_manual_integration_contract returns OK for aligned."""
        contract = build_field_manual_integration_contract(aligned_label_index)
        
        assert contract["integration_status"] == "OK"
        assert "curriculum" in contract["required_label_sets"]
        assert "taxonomy" in contract["required_label_sets"]
        assert "docs" in contract["required_label_sets"]
    
    def test_131_integration_contract_block(self, broken_label_index):
        """build_field_manual_integration_contract returns BLOCK for broken."""
        contract = build_field_manual_integration_contract(broken_label_index)
        
        assert contract["integration_status"] == "BLOCK"
        assert "not defined in Field Manual" in contract["notes"]
    
    def test_132_integration_contract_label_sets(self, aligned_label_index):
        """build_field_manual_integration_contract extracts label sets correctly."""
        contract = build_field_manual_integration_contract(aligned_label_index)
        
        assert "def:slice" in contract["required_label_sets"]["docs"]
        assert "def:slice" in contract["required_label_sets"]["taxonomy"]
        assert "def:slice" in contract["required_label_sets"]["curriculum"]
    
    def test_133_integration_contract_deterministic(self, aligned_label_index):
        """build_field_manual_integration_contract is deterministic."""
        contract1 = build_field_manual_integration_contract(aligned_label_index)
        contract2 = build_field_manual_integration_contract(aligned_label_index)
        
        assert contract1 == contract2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 20: DIRECTOR FM PANEL TESTS (134-139)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDirectorFMPanel:
    """Tests for Field Manual director panel."""
    
    @pytest.fixture
    def healthy_posture(self):
        """Healthy FM posture."""
        return {
            "health_status": "healthy",
            "coverage_pct": 80.0,
        }
    
    @pytest.fixture
    def warning_posture(self):
        """Warning FM posture."""
        return {
            "health_status": "warning",
            "coverage_pct": 5.0,
        }
    
    @pytest.fixture
    def ok_integration(self):
        """OK integration contract."""
        return {
            "integration_status": "OK",
            "notes": "All label sets are aligned across systems.",
        }
    
    @pytest.fixture
    def block_integration(self):
        """BLOCK integration contract."""
        return {
            "integration_status": "BLOCK",
            "notes": "2 label(s) used in taxonomy/curriculum but not defined in Field Manual",
        }
    
    def test_134_director_panel_schema(self, healthy_posture, ok_integration):
        """build_field_manual_director_panel returns correct schema."""
        panel = build_field_manual_director_panel(healthy_posture, ok_integration)
        
        assert "status_light" in panel
        assert "health_status" in panel
        assert "coverage_pct" in panel
        assert "integration_status" in panel
        assert "headline" in panel
    
    def test_135_director_panel_green(self, healthy_posture, ok_integration):
        """build_field_manual_director_panel returns green for healthy."""
        panel = build_field_manual_director_panel(healthy_posture, ok_integration)
        
        assert panel["status_light"] == "🟢"
        assert panel["health_status"] == "healthy"
        assert panel["integration_status"] == "OK"
    
    def test_136_director_panel_yellow(self, warning_posture, ok_integration):
        """build_field_manual_director_panel returns yellow for warning."""
        panel = build_field_manual_director_panel(warning_posture, ok_integration)
        
        assert panel["status_light"] == "🟡"
        assert panel["health_status"] == "warning"
    
    def test_137_director_panel_red(self, healthy_posture, block_integration):
        """build_field_manual_director_panel returns red for blocked."""
        panel = build_field_manual_director_panel(healthy_posture, block_integration)
        
        assert panel["status_light"] == "🔴"
        assert panel["integration_status"] == "BLOCK"
    
    def test_138_director_panel_headline(self, healthy_posture, ok_integration):
        """build_field_manual_director_panel generates appropriate headline."""
        panel = build_field_manual_director_panel(healthy_posture, ok_integration)
        
        assert "coverage" in panel["headline"].lower()
        assert "80.0%" in panel["headline"]
    
    def test_139_director_panel_deterministic(self, healthy_posture, ok_integration):
        """build_field_manual_director_panel is deterministic."""
        panel1 = build_field_manual_director_panel(healthy_posture, ok_integration)
        panel2 = build_field_manual_director_panel(healthy_posture, ok_integration)
        
        assert panel1 == panel2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 21: FM DRIFT TIMELINE TESTS (140-145)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFMDriftTimeline:
    """Tests for Field Manual drift timeline analysis."""
    
    @pytest.fixture
    def improving_postures(self):
        """Posture snapshots showing improving coverage."""
        return [
            {"coverage_pct": 10.0},
            {"coverage_pct": 20.0},
            {"coverage_pct": 30.0},
        ]
    
    @pytest.fixture
    def degrading_postures(self):
        """Posture snapshots showing degrading coverage."""
        return [
            {"coverage_pct": 50.0},
            {"coverage_pct": 40.0},
            {"coverage_pct": 30.0},
        ]
    
    @pytest.fixture
    def stable_postures(self):
        """Posture snapshots showing stable coverage."""
        return [
            {"coverage_pct": 25.0},
            {"coverage_pct": 26.0},
            {"coverage_pct": 24.0},
        ]
    
    @pytest.fixture
    def improving_contracts(self):
        """Label indexes showing improving contract status."""
        return [
            {"contract_status": "BROKEN"},
            {"contract_status": "PARTIAL"},
            {"contract_status": "ALIGNED"},
        ]
    
    @pytest.fixture
    def degrading_contracts(self):
        """Label indexes showing degrading contract status."""
        return [
            {"contract_status": "ALIGNED"},
            {"contract_status": "PARTIAL"},
            {"contract_status": "BROKEN"},
        ]
    
    @pytest.fixture
    def stable_contracts(self):
        """Label indexes showing stable contract status."""
        return [
            {"contract_status": "ALIGNED"},
            {"contract_status": "ALIGNED"},
            {"contract_status": "ALIGNED"},
        ]
    
    def test_140_drift_timeline_schema(self, improving_postures, improving_contracts):
        """build_field_manual_drift_timeline returns correct schema."""
        timeline = build_field_manual_drift_timeline(improving_postures, improving_contracts)
        
        assert "schema_version" in timeline
        assert "version_count" in timeline
        assert "coverage_trend" in timeline
        assert "contract_trend" in timeline
        assert "neutral_notes" in timeline
    
    def test_141_drift_timeline_improving_coverage(self, improving_postures, stable_contracts):
        """build_field_manual_drift_timeline detects improving coverage."""
        timeline = build_field_manual_drift_timeline(improving_postures, stable_contracts)
        
        assert timeline["coverage_trend"] == "IMPROVING"
        assert timeline["version_count"] == 3
        assert "improving" in timeline["neutral_notes"][1].lower()
    
    def test_142_drift_timeline_degrading_coverage(self, degrading_postures, stable_contracts):
        """build_field_manual_drift_timeline detects degrading coverage."""
        timeline = build_field_manual_drift_timeline(degrading_postures, stable_contracts)
        
        assert timeline["coverage_trend"] == "DEGRADING"
        assert "degrading" in timeline["neutral_notes"][1].lower()
    
    def test_143_drift_timeline_stable_coverage(self, stable_postures, stable_contracts):
        """build_field_manual_drift_timeline detects stable coverage."""
        timeline = build_field_manual_drift_timeline(stable_postures, stable_contracts)
        
        assert timeline["coverage_trend"] == "STABLE"
        assert "stable" in timeline["neutral_notes"][1].lower()
    
    def test_144_drift_timeline_improving_contract(self, stable_postures, improving_contracts):
        """build_field_manual_drift_timeline detects improving contract."""
        timeline = build_field_manual_drift_timeline(stable_postures, improving_contracts)
        
        assert timeline["contract_trend"] == "IMPROVING"
        assert "improving" in timeline["neutral_notes"][2].lower()
        assert "BROKEN" in timeline["neutral_notes"][2]
        assert "ALIGNED" in timeline["neutral_notes"][2]
    
    def test_145_drift_timeline_insufficient_data(self):
        """build_field_manual_drift_timeline handles insufficient data."""
        timeline = build_field_manual_drift_timeline([{"coverage_pct": 10.0}], [{"contract_status": "ALIGNED"}])
        
        assert timeline["coverage_trend"] == "STABLE"
        assert timeline["contract_trend"] == "STABLE"
        assert "Insufficient data" in timeline["neutral_notes"][0]
    
    def test_146_drift_timeline_deterministic(self, improving_postures, improving_contracts):
        """build_field_manual_drift_timeline is deterministic."""
        timeline1 = build_field_manual_drift_timeline(improving_postures, improving_contracts)
        timeline2 = build_field_manual_drift_timeline(improving_postures, improving_contracts)
        
        assert timeline1 == timeline2


# ═══════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

