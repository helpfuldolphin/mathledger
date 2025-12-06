#!/usr/bin/env python3
"""
Evidence Gate Linter - docs_lint
Enforces hash-linked citations on new documentation

Detects metric claims lacking [hash:xxxxxxxx] citations in git diffs.
Exit codes:
  0 - PASS: No violations
  1 - FAIL: Violations found
  2 - ABSTAIN: External doc excluded (not under docs/)
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

# Patterns that indicate empirical claims requiring hash citations
CLAIM_PATTERNS = [
    # Numeric metrics
    r'\d+\.?\d*\s*proofs?/(?:hour|sec|min)',
    r'\d+\.?\d*%\s*(?:improvement|uplift|increase|decrease|reduction)',
    r'\d+\.?\d*x\s*(?:speedup|faster|slower|uplift)',
    r'p[_-]?value[:\s=]+\d+\.?\d*',
    r'uplift[:\s=]+\d+\.?\d*',
    r'baseline[:\s=]+\d+\.?\d*',
    r'guided[:\s=]+\d+\.?\d*',
    
    # Status claims
    r'(?:PASS|FAIL|WARNING|ERROR)[:\s]+[^\n]+',
    r'Gate[:\s]+[^\n]+',
    r'verified[:\s]+[^\n]+',
    
    # Performance claims
    r'latency[:\s]+\d+',
    r'throughput[:\s]+\d+',
    r'coverage[:\s]+\d+',
]

# Hash citation pattern
HASH_CITATION_PATTERN = r'\[hash:[a-f0-9]{8}\]'

class ViolationDetector:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.violations = []
        
    def get_git_diff(self, base_ref: str = "HEAD") -> List[Tuple[str, List[str]]]:
        """Get git diff of added lines, grouped by file"""
        try:
            # Get diff with added lines only
            result = subprocess.run(
                ['git', 'diff', '--unified=0', base_ref, '--', 'docs/', '*.md'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return []
            
            return self._parse_diff(result.stdout)
        
        except Exception as e:
            print(f"Error getting git diff: {e}", file=sys.stderr)
            return []
    
    def _parse_diff(self, diff_output: str) -> List[Tuple[str, List[str]]]:
        """Parse git diff output into (filename, added_lines) tuples"""
        files_with_additions = []
        current_file = None
        added_lines = []
        
        for line in diff_output.split('\n'):
            # New file marker
            if line.startswith('diff --git'):
                if current_file and added_lines:
                    files_with_additions.append((current_file, added_lines))
                current_file = None
                added_lines = []
            
            # File path
            elif line.startswith('+++'):
                # Extract path after 'b/'
                match = re.search(r'\+\+\+ b/(.+)$', line)
                if match:
                    current_file = match.group(1)
            
            # Added line
            elif line.startswith('+') and not line.startswith('+++'):
                # Remove leading '+'
                added_lines.append(line[1:])
        
        # Don't forget last file
        if current_file and added_lines:
            files_with_additions.append((current_file, added_lines))
        
        return files_with_additions
    
    def is_docs_file(self, filepath: str) -> bool:
        """Check if file is under docs/ or is a root .md file"""
        return filepath.startswith('docs/') or (filepath.endswith('.md') and '/' not in filepath)
    
    def detect_claim(self, line: str) -> bool:
        """Check if line contains an empirical claim"""
        for pattern in CLAIM_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def has_hash_citation(self, line: str) -> bool:
        """Check if line contains a hash citation"""
        return bool(re.search(HASH_CITATION_PATTERN, line))
    
    def lint_file(self, filepath: str, added_lines: List[str]) -> List[Dict]:
        """Lint a single file for violations"""
        file_violations = []
        
        for line_num, line in enumerate(added_lines, start=1):
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                continue
            
            # Check if line contains a claim
            if self.detect_claim(line):
                # Check if it has a hash citation
                if not self.has_hash_citation(line):
                    file_violations.append({
                        'file': filepath,
                        'line': line_num,
                        'content': line.strip(),
                        'type': 'missing_hash_citation'
                    })
        
        return file_violations
    
    def lint_diff(self, base_ref: str = "HEAD") -> int:
        """Lint git diff for violations"""
        diff_files = self.get_git_diff(base_ref)
        
        if not diff_files:
            print("[ABSTAIN] No documentation changes detected")
            return 2
        
        # Check if any files are docs
        docs_files = [(f, lines) for f, lines in diff_files if self.is_docs_file(f)]
        
        if not docs_files:
            print("[ABSTAIN] External doc excluded (not under docs/)")
            return 2
        
        # Lint each docs file
        for filepath, added_lines in docs_files:
            violations = self.lint_file(filepath, added_lines)
            self.violations.extend(violations)
        
        # Report results
        if not self.violations:
            print("[PASS] Evidence Graph Lint: 0 violations")
            return 0
        else:
            print(f"[FAIL] Evidence Graph Lint: {len(self.violations)} violation(s) found\n")
            for v in self.violations:
                print(f"  {v['file']}:{v['line']}")
                print(f"    Claim: {v['content']}")
                print(f"    Missing: [hash:xxxxxxxx] citation\n")
            return 1

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evidence Gate Linter - Enforce hash-linked citations'
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
    
    args = parser.parse_args()
    
    detector = ViolationDetector(args.repo)
    exit_code = detector.lint_diff(args.base)
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()

