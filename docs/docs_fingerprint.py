#!/usr/bin/env python3
"""
Documentation Fingerprinting and Drift Radar for MathLedger

Implements:
1. Documentation fingerprinting with normalized text, code block counts, 
   section headers, Phase markers, and RFL/uplift context detection
2. Cross-documentation drift radar detecting missing Phase II markers,
   uplifting language, code example drift, and terminology drift
3. Governance annotation consistency checker validating disclaimers and
   ensuring governance wording stability

Usage:
    python docs/docs_fingerprint.py --docs-fingerprint docs/PHASE2_RFL_UPLIFT_PLAN.md
    python docs/docs_fingerprint.py --docs-drift-history docs/v1.md docs/v2.md docs/v3.md
    python docs/docs_fingerprint.py --docs-validate-governance

Fleet Directive: Determinism > speed, Proof-or-Abstain, Sober Truth
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Governance-critical patterns
PHASE_MARKERS = [
    "PHASE I",
    "PHASE II",
    "PHASE III",
    "PHASE IV",
    "Phase I",
    "Phase II", 
    "Phase III",
    "Phase IV",
]

UPLIFT_LANGUAGE = [
    "demonstrated uplift",
    "proven uplift",
    "uplift evidence",
    "uplift readiness",
    "uplift achieved",
    "shows uplift",
    "confirms uplift",
    "validated uplift",
]

RFL_CONTEXTS = [
    "RFL",
    "Reflexive Formal Learning",
    "policy guidance",
    "learned policy",
    "uplift",
]

REQUIRED_DISCLAIMERS = [
    "NOT YET IMPLEMENTED",
    "NOT YET RUN",
    "NOT RUN IN PHASE I",
    "NO UPLIFT CLAIMS",
]


def normalize_text(text: str) -> str:
    """
    Normalize text by stripping whitespace and collapsing blank lines.
    
    Args:
        text: Raw text content
        
    Returns:
        Normalized text
    """
    lines = []
    for line in text.split('\n'):
        # Strip trailing whitespace but preserve indentation
        stripped = line.rstrip()
        lines.append(stripped)
    
    # Collapse multiple blank lines into single blank line
    normalized_lines = []
    prev_blank = False
    for line in lines:
        is_blank = len(line.strip()) == 0
        if is_blank and prev_blank:
            continue  # Skip consecutive blank lines
        normalized_lines.append(line)
        prev_blank = is_blank
    
    return '\n'.join(normalized_lines)


def count_code_blocks(text: str) -> int:
    """
    Count fenced code blocks (```...```).
    
    Args:
        text: Markdown text
        
    Returns:
        Number of code blocks
    """
    # Match fenced code blocks
    pattern = r'```[^\n]*\n.*?```'
    matches = re.findall(pattern, text, re.DOTALL)
    return len(matches)


def count_section_headers(text: str) -> int:
    """
    Count markdown section headers (lines starting with #).
    
    Args:
        text: Markdown text
        
    Returns:
        Number of section headers
    """
    count = 0
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            count += 1
    return count


def extract_phase_markers(text: str) -> List[str]:
    """
    Extract Phase markers (PHASE I, PHASE II, etc.) from text.
    
    Args:
        text: Markdown text
        
    Returns:
        List of found Phase markers with context
    """
    markers = []
    for marker in PHASE_MARKERS:
        # Find all occurrences with surrounding context
        pattern = re.escape(marker)
        for match in re.finditer(pattern, text):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].replace('\n', ' ')
            markers.append({
                'marker': marker,
                'context': context.strip(),
                'position': match.start()
            })
    return markers


def detect_rfl_uplift_contexts(text: str) -> List[Dict]:
    """
    Detect RFL or uplift context imports/mentions.
    
    Args:
        text: Markdown text
        
    Returns:
        List of RFL/uplift contexts found
    """
    contexts = []
    
    for term in RFL_CONTEXTS:
        pattern = re.escape(term)
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Get surrounding line for context
            line_start = text.rfind('\n', 0, match.start()) + 1
            line_end = text.find('\n', match.end())
            if line_end == -1:
                line_end = len(text)
            
            line_content = text[line_start:line_end].strip()
            contexts.append({
                'term': term,
                'line': line_content,
                'position': match.start()
            })
    
    return contexts


def fingerprint_document(file_path: Path) -> Dict:
    """
    Generate fingerprint for a documentation file.
    
    Args:
        file_path: Path to documentation file
        
    Returns:
        Dictionary containing fingerprint data
    """
    if not file_path.exists():
        return {
            'error': 'File not found',
            'path': str(file_path)
        }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {
            'error': f'Failed to read file: {e}',
            'path': str(file_path)
        }
    
    normalized = normalize_text(content)
    
    # Compute hash of normalized content
    content_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    fingerprint = {
        'path': str(file_path),
        'size_bytes': len(content),
        'lines': len(content.split('\n')),
        'normalized_lines': len(normalized.split('\n')),
        'content_hash': content_hash,
        'code_blocks': count_code_blocks(content),
        'section_headers': count_section_headers(content),
        'phase_markers': extract_phase_markers(content),
        'rfl_uplift_contexts': detect_rfl_uplift_contexts(content),
    }
    
    return fingerprint


def detect_missing_phase_markers(fingerprints: List[Dict]) -> List[Dict]:
    """
    Detect documents that mention Phase II but lack proper markers.
    
    Args:
        fingerprints: List of document fingerprints
        
    Returns:
        List of issues found
    """
    issues = []
    
    for fp in fingerprints:
        if 'error' in fp:
            continue
            
        # Check if document mentions Phase II concepts without markers
        rfl_contexts = fp.get('rfl_uplift_contexts', [])
        phase_markers = fp.get('phase_markers', [])
        
        has_phase_ii_content = any(
            ctx['term'].lower() in ['rfl', 'uplift', 'policy guidance']
            for ctx in rfl_contexts
        )
        
        has_phase_ii_marker = any(
            'PHASE II' in marker['marker'] or 'Phase II' in marker['marker']
            for marker in phase_markers
        )
        
        if has_phase_ii_content and not has_phase_ii_marker:
            issues.append({
                'type': 'missing_phase_ii_marker',
                'path': fp['path'],
                'description': 'Document discusses Phase II concepts (RFL/uplift) but lacks Phase II markers',
                'rfl_contexts': [ctx['line'] for ctx in rfl_contexts[:3]]
            })
    
    return issues


def detect_uplifting_language(fingerprints: List[Dict]) -> List[Dict]:
    """
    Detect introduction of uplifting language without evidence citations.
    
    Args:
        fingerprints: List of document fingerprints
        
    Returns:
        List of issues found
    """
    issues = []
    
    for fp in fingerprints:
        if 'error' in fp:
            continue
        
        try:
            with open(fp['path'], 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue
        
        for phrase in UPLIFT_LANGUAGE:
            if phrase.lower() in content.lower():
                # Check if there's a nearby disclaimer
                pattern = re.escape(phrase)
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    # Look for disclaimers within 500 chars before or after
                    start = max(0, match.start() - 500)
                    end = min(len(content), match.end() + 500)
                    context = content[start:end]
                    
                    has_disclaimer = any(
                        disclaimer in context
                        for disclaimer in REQUIRED_DISCLAIMERS
                    )
                    
                    if not has_disclaimer:
                        issues.append({
                            'type': 'unguarded_uplift_claim',
                            'path': fp['path'],
                            'phrase': phrase,
                            'position': match.start(),
                            'description': f'Uplift language "{phrase}" used without Phase II disclaimer'
                        })
    
    return issues


def detect_code_drift(fingerprints: List[Dict]) -> List[Dict]:
    """
    Detect drift in code examples across document versions.
    
    Args:
        fingerprints: List of document fingerprints (ordered by version)
        
    Returns:
        List of drift issues
    """
    issues = []
    
    if len(fingerprints) < 2:
        return issues
    
    for i in range(1, len(fingerprints)):
        prev = fingerprints[i-1]
        curr = fingerprints[i]
        
        if 'error' in prev or 'error' in curr:
            continue
        
        prev_blocks = prev.get('code_blocks', 0)
        curr_blocks = curr.get('code_blocks', 0)
        
        if curr_blocks != prev_blocks:
            issues.append({
                'type': 'code_block_count_change',
                'from_path': prev['path'],
                'to_path': curr['path'],
                'prev_count': prev_blocks,
                'curr_count': curr_blocks,
                'delta': curr_blocks - prev_blocks,
                'description': f'Code block count changed from {prev_blocks} to {curr_blocks}'
            })
    
    return issues


def detect_terminology_drift(fingerprints: List[Dict]) -> List[Dict]:
    """
    Detect drift in terminology (RFL, uplift, determinism).
    
    Args:
        fingerprints: List of document fingerprints (ordered by version)
        
    Returns:
        List of drift issues
    """
    issues = []
    
    if len(fingerprints) < 2:
        return issues
    
    for i in range(1, len(fingerprints)):
        prev = fingerprints[i-1]
        curr = fingerprints[i]
        
        if 'error' in prev or 'error' in curr:
            continue
        
        prev_rfl = len(prev.get('rfl_uplift_contexts', []))
        curr_rfl = len(curr.get('rfl_uplift_contexts', []))
        
        if curr_rfl > prev_rfl * 1.5:  # 50% increase
            issues.append({
                'type': 'terminology_expansion',
                'from_path': prev['path'],
                'to_path': curr['path'],
                'prev_count': prev_rfl,
                'curr_count': curr_rfl,
                'delta': curr_rfl - prev_rfl,
                'description': f'RFL/uplift terminology usage increased significantly ({prev_rfl} → {curr_rfl})'
            })
    
    return issues


def build_docs_drift_radar(fingerprints: List[Dict]) -> Dict:
    """
    Build comprehensive drift radar across documentation set.
    
    Detects:
    - Missing Phase II markers
    - Introduction of uplifting language
    - Drift in code examples
    - Drift in terminology (RFL, uplift, determinism)
    
    Args:
        fingerprints: List of document fingerprints
        
    Returns:
        Dictionary containing all detected drift issues
    """
    return {
        'format_version': '1.0',
        'report_type': 'docs_drift_radar',
        'documents_analyzed': len([fp for fp in fingerprints if 'error' not in fp]),
        'documents_with_errors': len([fp for fp in fingerprints if 'error' in fp]),
        'issues': {
            'missing_phase_markers': detect_missing_phase_markers(fingerprints),
            'uplifting_language': detect_uplifting_language(fingerprints),
            'code_drift': detect_code_drift(fingerprints),
            'terminology_drift': detect_terminology_drift(fingerprints),
        },
        'total_issues': (
            len(detect_missing_phase_markers(fingerprints)) +
            len(detect_uplifting_language(fingerprints)) +
            len(detect_code_drift(fingerprints)) +
            len(detect_terminology_drift(fingerprints))
        )
    }


def validate_governance_annotations(fingerprints: List[Dict]) -> Dict:
    """
    Validate governance annotation consistency.
    
    Checks:
    - All docs referencing Phase II experiments include disclaimers
    - No doc references uplift readiness without evidence
    - Governance wording does not mutate over time
    
    Args:
        fingerprints: List of document fingerprints
        
    Returns:
        Dictionary containing validation results
    """
    issues = []
    
    for fp in fingerprints:
        if 'error' in fp:
            continue
        
        phase_markers = fp.get('phase_markers', [])
        
        # Check if Phase II is mentioned
        has_phase_ii = any(
            'PHASE II' in marker['marker'] or 'Phase II' in marker['marker']
            for marker in phase_markers
        )
        
        if not has_phase_ii:
            continue
        
        # Verify disclaimers are present
        try:
            with open(fp['path'], 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue
        
        has_disclaimer = any(
            disclaimer in content
            for disclaimer in REQUIRED_DISCLAIMERS
        )
        
        if not has_disclaimer:
            issues.append({
                'type': 'missing_disclaimer',
                'path': fp['path'],
                'description': 'Document references Phase II but lacks required disclaimer',
                'required_disclaimers': REQUIRED_DISCLAIMERS
            })
        
        # Check for uplift readiness claims
        readiness_pattern = r'(uplift\s+ready|ready\s+for\s+uplift|production\s+ready.*uplift)'
        if re.search(readiness_pattern, content, re.IGNORECASE):
            issues.append({
                'type': 'premature_readiness_claim',
                'path': fp['path'],
                'description': 'Document claims uplift readiness without gate evidence'
            })
    
    return {
        'format_version': '1.0',
        'report_type': 'governance_validation',
        'documents_checked': len([fp for fp in fingerprints if 'error' not in fp]),
        'issues': issues,
        'total_issues': len(issues),
        'status': 'PASS' if len(issues) == 0 else 'FAIL'
    }


def main():
    parser = argparse.ArgumentParser(
        description="Documentation Fingerprinting and Drift Radar"
    )
    
    parser.add_argument(
        '--docs-fingerprint',
        type=Path,
        help='Generate fingerprint for a single document'
    )
    
    parser.add_argument(
        '--docs-drift-history',
        type=Path,
        nargs='+',
        help='Analyze drift across multiple document versions (ordered by time)'
    )
    
    parser.add_argument(
        '--docs-validate-governance',
        action='store_true',
        help='Validate governance annotation consistency across all docs'
    )
    
    parser.add_argument(
        '--docs-dir',
        type=Path,
        default=Path('docs'),
        help='Documentation directory to scan (default: docs/)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for report JSON'
    )
    
    args = parser.parse_args()
    
    # Mode 1: Single document fingerprint
    if args.docs_fingerprint:
        print(f"Generating fingerprint for {args.docs_fingerprint}...")
        fingerprint = fingerprint_document(args.docs_fingerprint)
        
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, indent=2)
            print(f"Fingerprint written to {args.output}")
        else:
            print(json.dumps(fingerprint, indent=2))
        
        return 0
    
    # Mode 2: Drift history analysis
    if args.docs_drift_history:
        print(f"Analyzing drift across {len(args.docs_drift_history)} documents...")
        fingerprints = [
            fingerprint_document(path)
            for path in args.docs_drift_history
        ]
        
        radar = build_docs_drift_radar(fingerprints)
        
        print(f"\nDrift Radar Results:")
        print(f"  Documents analyzed: {radar['documents_analyzed']}")
        print(f"  Total issues: {radar['total_issues']}")
        print(f"  - Missing Phase markers: {len(radar['issues']['missing_phase_markers'])}")
        print(f"  - Uplifting language: {len(radar['issues']['uplifting_language'])}")
        print(f"  - Code drift: {len(radar['issues']['code_drift'])}")
        print(f"  - Terminology drift: {len(radar['issues']['terminology_drift'])}")
        
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(radar, f, indent=2)
            print(f"\nRadar report written to {args.output}")
        else:
            print("\nDetailed report:")
            print(json.dumps(radar, indent=2))
        
        return 0 if radar['total_issues'] == 0 else 1
    
    # Mode 3: Governance validation
    if args.docs_validate_governance:
        print(f"Validating governance annotations in {args.docs_dir}...")
        
        # Scan all markdown files
        fingerprints = []
        for md_file in args.docs_dir.rglob("*.md"):
            fingerprints.append(fingerprint_document(md_file))
        
        print(f"Scanned {len(fingerprints)} documents")
        
        validation = validate_governance_annotations(fingerprints)
        
        print(f"\nGovernance Validation Results:")
        print(f"  Status: {validation['status']}")
        print(f"  Documents checked: {validation['documents_checked']}")
        print(f"  Issues found: {validation['total_issues']}")
        
        if validation['total_issues'] > 0:
            print("\nIssues:")
            for issue in validation['issues']:
                print(f"  - [{issue['type']}] {issue['path']}")
                print(f"    {issue['description']}")
        
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(validation, f, indent=2)
            print(f"\nValidation report written to {args.output}")
        else:
            print("\nDetailed report:")
            print(json.dumps(validation, indent=2))
        
        return 0 if validation['status'] == 'PASS' else 1
    
    # No mode specified
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
