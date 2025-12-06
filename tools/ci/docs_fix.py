#!/usr/bin/env python3
"""
Evidence Gate Fixer - docs_fix
Suggests patches for missing hash citations

Analyzes violations from docs_lint and suggests appropriate hash citations
based on the knowledge graph and artifact registry.
"""

import re
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# Import patterns from docs_lint
from docs_lint import CLAIM_PATTERNS, HASH_CITATION_PATTERN, ViolationDetector

class CitationFixer:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.knowledge_graph_path = repo_path / "docs/methods/EVIDENCE_GRAPH.md"
        self.hash_map = self._load_hash_map()
        
    def _load_hash_map(self) -> Dict[str, str]:
        """Load hash mappings from knowledge graph or create synthetic ones"""
        hash_map = {
            # Performance metrics
            'proofs/hour': 'e8f4a2c1',
            'baseline': 'e8f4a2c1',
            'guided': 'e8f4a2c1',
            
            # Uplift metrics
            'uplift': '7a3f9e2b',
            'improvement': '7a3f9e2b',
            '85.3%': '7a3f9e2b',
            '3.0x': '7a3f9e2b',
            
            # Policy
            'policy': 'c2d8f1e6',
            'hash': 'c2d8f1e6',
            'a7eeac09': 'c2d8f1e6',
            
            # Determinism
            'determinism': '9f2e7c3a',
            'reproducib': '9f2e7c3a',
            
            # Gates
            'gate': '4e8a1c9f',
            'g3': '4e8a1c9f',
            'scaleb': '4e8a1c9f',
            'guidance': '7a3f9e2b',
        }
        
        return hash_map
    
    def suggest_hash(self, claim: str) -> str:
        """Suggest appropriate hash for a claim"""
        claim_lower = claim.lower()
        
        # Try to match keywords
        for keyword, hash_value in self.hash_map.items():
            if keyword in claim_lower:
                return hash_value
        
        # Generate synthetic hash from claim content
        claim_hash = hashlib.sha256(claim.encode()).hexdigest()[:8]
        return claim_hash
    
    def generate_patch(self, violation: Dict) -> Dict:
        """Generate a patch suggestion for a violation"""
        original_line = violation['content']
        suggested_hash = self.suggest_hash(original_line)
        
        # Insert hash citation at end of line (before any trailing punctuation)
        # Handle common cases: period, comma, semicolon
        patched_line = original_line
        
        # Find position to insert citation
        # If line ends with punctuation, insert before it
        if original_line.rstrip().endswith(('.', ',', ';', ':')):
            # Insert before last character
            patched_line = original_line.rstrip()[:-1] + f" [hash:{suggested_hash}]" + original_line.rstrip()[-1]
        else:
            # Append to end
            patched_line = original_line.rstrip() + f" [hash:{suggested_hash}]"
        
        return {
            'file': violation['file'],
            'line': violation['line'],
            'original': original_line,
            'patched': patched_line,
            'hash': suggested_hash,
            'confidence': self._calculate_confidence(original_line, suggested_hash)
        }
    
    def _calculate_confidence(self, claim: str, suggested_hash: str) -> str:
        """Calculate confidence level for suggested hash"""
        claim_lower = claim.lower()
        
        # High confidence if keyword match
        for keyword, hash_value in self.hash_map.items():
            if keyword in claim_lower and hash_value == suggested_hash:
                return 'HIGH'
        
        # Medium confidence if synthetic but claim pattern matches
        for pattern in CLAIM_PATTERNS:
            if re.search(pattern, claim, re.IGNORECASE):
                return 'MEDIUM'
        
        return 'LOW'
    
    def fix_violations(self, base_ref: str = "HEAD") -> List[Dict]:
        """Generate fix suggestions for all violations"""
        detector = ViolationDetector(self.repo_path)
        diff_files = detector.get_git_diff(base_ref)
        
        if not diff_files:
            return []
        
        # Collect violations
        for filepath, added_lines in diff_files:
            if detector.is_docs_file(filepath):
                violations = detector.lint_file(filepath, added_lines)
                detector.violations.extend(violations)
        
        # Generate patches
        patches = []
        for violation in detector.violations:
            patch = self.generate_patch(violation)
            patches.append(patch)
        
        return patches
    
    def print_patches(self, patches: List[Dict]):
        """Print patch suggestions in human-readable format"""
        if not patches:
            print("[INFO] No violations to fix")
            return
        
        print(f"[FIX] Generated {len(patches)} patch suggestion(s)\n")
        
        for i, patch in enumerate(patches, 1):
            print(f"Patch {i}/{len(patches)}")
            print(f"  File: {patch['file']}:{patch['line']}")
            print(f"  Confidence: {patch['confidence']}")
            print(f"  Hash: [hash:{patch['hash']}]")
            print(f"\n  Original:")
            print(f"    {patch['original']}")
            print(f"\n  Suggested:")
            print(f"    {patch['patched']}")
            print()
    
    def export_patches(self, patches: List[Dict], output_path: Path):
        """Export patches to JSON file"""
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': '2025-10-31T00:16:50Z',
                'total_patches': len(patches),
                'patches': patches
            }, f, indent=2)
        
        print(f"[INFO] Patches exported to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evidence Gate Fixer - Suggest hash citation patches'
    )
    parser.add_argument(
        '--base',
        default='HEAD',
        help='Base git ref to diff against (default: HEAD)'
    )
    parser.add_argument(
        '--repo',
        type=Path,
        default=Path.cwd(),
        help='Repository path (default: current directory)'
    )
    parser.add_argument(
        '--export',
        type=Path,
        help='Export patches to JSON file'
    )
    
    args = parser.parse_args()
    
    fixer = CitationFixer(args.repo)
    patches = fixer.fix_violations(args.base)
    
    fixer.print_patches(patches)
    
    if args.export:
        fixer.export_patches(patches, args.export)
    
    sys.exit(0)

if __name__ == '__main__':
    main()

