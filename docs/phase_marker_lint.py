#!/usr/bin/env python3
"""
Automatic Phase Markers Consistency Check

Ensures every relevant doc clearly indicates Phase I vs Phase II and never
accidentally claims uplift without appropriate disclaimers.

Usage:
    python docs/phase_marker_lint.py [--config CONFIG] [--verbose]

Examples:
    python docs/phase_marker_lint.py
    python docs/phase_marker_lint.py --config docs/phase_marker_rules.yaml
    python docs/phase_marker_lint.py --verbose
"""

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Any

import yaml


class PhaseMarkerLinter:
    """Validates Phase markers and uplift disclaimers in documentation."""
    
    def __init__(self, root_dir: Path, verbose: bool = False):
        self.root_dir = root_dir
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def log(self, msg: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {msg}")
    
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        content = config_path.read_text(encoding='utf-8')
        return yaml.safe_load(content)
    
    def check_phase_marker(self, content: str, filepath: str, required_phase: str) -> bool:
        """
        Check if document has required Phase marker.
        
        Args:
            content: Document content
            filepath: Path to document (for error reporting)
            required_phase: Required phase marker (e.g., "PHASE II")
        
        Returns:
            True if marker is present, False otherwise
        """
        # Look for phase markers in various formats
        patterns = [
            rf'\*\*STATUS:\s*{re.escape(required_phase)}',  # **STATUS: PHASE II**
            rf'>\s*\*\*STATUS:\s*{re.escape(required_phase)}',  # > **STATUS: PHASE II**
            rf'#.*{re.escape(required_phase)}',  # # PHASE II - Title
            rf'{re.escape(required_phase)}\s*—',  # PHASE II — subtitle
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.log(f"Found phase marker '{required_phase}' in {filepath}")
                return True
        
        return False
    
    def check_uplift_disclaimer(self, content: str, filepath: str) -> bool:
        """
        Check if document mentioning uplift has proper disclaimer.
        
        Returns:
            True if disclaimer is present, False otherwise
        """
        # Patterns that indicate uplift is mentioned
        uplift_patterns = [
            r'\buplift\b',
            r'\bdemonstrated.*improvement\b',
            r'\bresults.*show.*improvement\b',
        ]
        
        has_uplift_mention = False
        for pattern in uplift_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                has_uplift_mention = True
                break
        
        if not has_uplift_mention:
            self.log(f"No uplift mention in {filepath}, no disclaimer needed")
            return True
        
        # Patterns that indicate proper disclaimer
        disclaimer_patterns = [
            r'NO\s+UPLIFT\s+CLAIMS\s+MAY\s+BE\s+MADE',
            r'no\s+empirical\s+uplift.*yet',
            r'no\s+uplift.*demonstrated',
            r'no\s+uplift.*observed',
            r'uplift.*has\s+not\s+been\s+demonstrated',
            r'NOT\s+YET\s+RUN',
            r'NOT\s+RUN\s+IN\s+PHASE\s+I',
        ]
        
        for pattern in disclaimer_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.log(f"Found uplift disclaimer in {filepath}")
                return True
        
        self.log(f"Uplift mentioned but no disclaimer in {filepath}")
        return False
    
    def check_forbidden_claims(self, content: str, filepath: str, forbidden_claims: List[str]) -> List[str]:
        """
        Check if document contains forbidden claims.
        
        Returns:
            List of found forbidden claims
        """
        found_claims = []
        
        for claim in forbidden_claims:
            # Create case-insensitive pattern
            pattern = re.escape(claim)
            if re.search(pattern, content, re.IGNORECASE):
                found_claims.append(claim)
                self.log(f"Found forbidden claim '{claim}' in {filepath}")
        
        return found_claims
    
    def check_required_sections(self, content: str, filepath: str, required_sections: List[str]) -> List[str]:
        """
        Check if document contains required sections.
        
        Returns:
            List of missing sections
        """
        missing = []
        
        for section in required_sections:
            # Look for markdown headers
            patterns = [
                rf'^#\s+{re.escape(section)}',
                rf'^##\s+{re.escape(section)}',
                rf'^###\s+{re.escape(section)}',
            ]
            
            found = False
            for pattern in patterns:
                if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                    found = True
                    break
            
            if not found:
                missing.append(section)
        
        return missing
    
    def check_file(self, filepath: Path, rules: Dict[str, Any]) -> bool:
        """
        Check a single file against phase marker rules.
        
        Returns:
            True if all checks pass, False otherwise
        """
        self.log(f"Checking {filepath}")
        
        try:
            content = filepath.read_text(encoding='utf-8')
        except Exception as e:
            self.errors.append(f"{filepath}: Failed to read file: {e}")
            return False
        
        all_valid = True
        
        # Check phase marker
        if 'phase_marker' in rules:
            required_phase = rules['phase_marker']
            if not self.check_phase_marker(content, str(filepath), required_phase):
                self.errors.append(
                    f"{filepath}: Missing required phase marker: {required_phase}"
                )
                all_valid = False
        
        # Check uplift disclaimer
        if rules.get('require_uplift_disclaimer', False):
            if not self.check_uplift_disclaimer(content, str(filepath)):
                self.errors.append(
                    f"{filepath}: Document mentions uplift but lacks required disclaimer"
                )
                all_valid = False
        
        # Check forbidden claims
        if 'forbidden_claims' in rules:
            forbidden = self.check_forbidden_claims(
                content, str(filepath), rules['forbidden_claims']
            )
            if forbidden:
                for claim in forbidden:
                    self.errors.append(
                        f"{filepath}: Contains forbidden claim: '{claim}'"
                    )
                all_valid = False
        
        # Check required sections
        if 'required_sections' in rules:
            missing = self.check_required_sections(
                content, str(filepath), rules['required_sections']
            )
            if missing:
                for section in missing:
                    self.warnings.append(
                        f"{filepath}: Missing recommended section: '{section}'"
                    )
        
        return all_valid
    
    def lint(self, config: Dict[str, Any]) -> int:
        """
        Run linting based on configuration.
        
        Returns:
            Exit code: 0 if all checks pass, 1 otherwise
        """
        rules_by_file = config.get('rules', {})
        
        if not rules_by_file:
            print("Warning: No rules defined in config", file=sys.stderr)
            return 0
        
        all_valid = True
        checked_count = 0
        
        for pattern, rules in rules_by_file.items():
            # Find matching files
            matches = glob.glob(str(self.root_dir / pattern), recursive=True)
            
            if not matches:
                self.warnings.append(f"No files matched pattern: {pattern}")
                continue
            
            for filepath in matches:
                filepath = Path(filepath)
                if not self.check_file(filepath, rules):
                    all_valid = False
                checked_count += 1
        
        # Report results
        print(f"\nChecked {checked_count} file(s)")
        
        if self.warnings:
            print(f"\n⚠️  Found {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} error(s):", file=sys.stderr)
            for error in self.errors:
                print(f"  {error}", file=sys.stderr)
            return 1
        else:
            print("\n✅ All phase marker checks passed!")
            return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Phase markers and uplift disclaimers in documentation"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('docs/phase_marker_rules.yaml'),
        help='Path to rules config file (default: docs/phase_marker_rules.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Find repository root
    root_dir = Path(__file__).parent.parent
    
    # Resolve config path
    config_path = root_dir / args.config
    
    linter = PhaseMarkerLinter(root_dir, verbose=args.verbose)
    
    try:
        config = linter.load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
    
    exit_code = linter.lint(config)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
