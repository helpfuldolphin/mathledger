#!/usr/bin/env python3
"""
Snippet Extractor for CLI Examples (Docs ↔ Code Sync)

Scans designated documentation files for CLI examples in bash code blocks
and validates that referenced scripts/modules exist and CLI arguments are
syntactically valid.

Usage:
    python docs/snippet_check.py [--docs PATTERN] [--verbose]

Examples:
    python docs/snippet_check.py
    python docs/snippet_check.py --docs "docs/PHASE2*.md"
    python docs/snippet_check.py --verbose
"""

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


class SnippetChecker:
    """Validates CLI examples in documentation files."""
    
    def __init__(self, root_dir: Path, verbose: bool = False):
        self.root_dir = root_dir
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def log(self, msg: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {msg}")
    
    def extract_bash_blocks(self, content: str, filepath: str) -> List[Tuple[int, str]]:
        """
        Extract bash code blocks from markdown content.
        
        Returns:
            List of tuples (line_number, code_block_content)
        """
        blocks = []
        lines = content.split('\n')
        in_bash_block = False
        current_block = []
        block_start_line = 0
        
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('```bash'):
                in_bash_block = True
                block_start_line = i
                current_block = []
            elif line.strip() == '```' and in_bash_block:
                in_bash_block = False
                if current_block:
                    blocks.append((block_start_line, '\n'.join(current_block)))
                current_block = []
            elif in_bash_block:
                current_block.append(line)
        
        return blocks
    
    def should_skip_snippet(self, code: str) -> bool:
        """Check if snippet should be skipped based on markers."""
        return '# DOCTEST: SKIP' in code
    
    def extract_commands(self, code: str) -> List[str]:
        """
        Extract commands from bash code block.
        Handles multi-line commands with backslash continuation.
        """
        commands = []
        current_cmd = []
        
        for line in code.split('\n'):
            # Skip comments and empty lines
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Handle line continuation
            if current_cmd or not line.startswith(' '):
                # Remove trailing backslash and leading spaces
                clean_line = line.rstrip('\\').strip()
                if clean_line:
                    current_cmd.append(clean_line)
                
                # If no backslash, command is complete
                if not line.rstrip().endswith('\\'):
                    if current_cmd:
                        commands.append(' '.join(current_cmd))
                        current_cmd = []
        
        # Handle any remaining command
        if current_cmd:
            commands.append(' '.join(current_cmd))
        
        return commands
    
    def is_relevant_command(self, cmd: str) -> bool:
        """Check if command is a relevant Python CLI command to validate."""
        # Check for python commands with experiments or module execution
        patterns = [
            r'python\s+experiments/',
            r'python\s+-m\s+',
            r'uv\s+run\s+python\s+experiments/',
            r'uv\s+run\s+python\s+-m\s+',
        ]
        return any(re.search(pattern, cmd) for pattern in patterns)
    
    def validate_command(self, cmd: str, filepath: str, line_num: int) -> bool:
        """
        Validate that a command references existing scripts.
        Returns True if valid, False otherwise.
        """
        valid = True
        
        # Extract the script path from various command formats
        # Handle: python experiments/script.py
        # Handle: python -m module.path
        # Handle: uv run python experiments/script.py
        
        # Remove uv run prefix if present
        cmd_clean = re.sub(r'^uv\s+run\s+', '', cmd)
        
        # Check for script file reference
        script_match = re.search(r'python\s+([^\s]+\.py)', cmd_clean)
        if script_match:
            script_path = script_match.group(1)
            full_path = self.root_dir / script_path
            
            if not full_path.exists():
                self.errors.append(
                    f"{filepath}:{line_num}: Referenced script does not exist: {script_path}"
                )
                valid = False
            else:
                self.log(f"✓ Script exists: {script_path}")
        
        # Check for module reference
        module_match = re.search(r'python\s+-m\s+([^\s]+)', cmd_clean)
        if module_match:
            module_path = module_match.group(1)
            # Convert module.path to file path
            module_file = module_path.replace('.', '/') + '.py'
            full_path = self.root_dir / module_file
            
            # Also check if it's a package with __init__.py
            package_path = self.root_dir / module_path.replace('.', '/')
            package_init = package_path / '__init__.py'
            
            if not full_path.exists() and not package_init.exists():
                self.errors.append(
                    f"{filepath}:{line_num}: Referenced module does not exist: {module_path}"
                )
                valid = False
            else:
                self.log(f"✓ Module exists: {module_path}")
        
        # Basic syntax validation - check for unmatched quotes
        quote_chars = ["'", '"']
        for quote in quote_chars:
            if cmd.count(quote) % 2 != 0:
                self.errors.append(
                    f"{filepath}:{line_num}: Unmatched quote in command: {quote}"
                )
                valid = False
        
        return valid
    
    def check_file(self, filepath: Path) -> bool:
        """
        Check a single documentation file for valid CLI snippets.
        Returns True if all snippets are valid.
        """
        self.log(f"Checking {filepath}")
        
        try:
            content = filepath.read_text(encoding='utf-8')
        except Exception as e:
            self.errors.append(f"{filepath}: Failed to read file: {e}")
            return False
        
        blocks = self.extract_bash_blocks(content, str(filepath))
        
        if not blocks:
            self.log(f"No bash blocks found in {filepath}")
            return True
        
        self.log(f"Found {len(blocks)} bash block(s) in {filepath}")
        
        all_valid = True
        for line_num, code in blocks:
            if self.should_skip_snippet(code):
                self.log(f"Skipping block at line {line_num} (DOCTEST: SKIP marker)")
                continue
            
            commands = self.extract_commands(code)
            for cmd in commands:
                if self.is_relevant_command(cmd):
                    self.log(f"Validating command: {cmd[:60]}...")
                    if not self.validate_command(cmd, str(filepath), line_num):
                        all_valid = False
        
        return all_valid
    
    def check_docs(self, pattern: str = "docs/PHASE2*.md") -> int:
        """
        Check documentation files matching the pattern.
        Returns exit code: 0 if all valid, 1 otherwise.
        """
        # Find matching files
        files = glob.glob(str(self.root_dir / pattern))
        
        if not files:
            print(f"Warning: No files matched pattern: {pattern}", file=sys.stderr)
            return 0
        
        print(f"Checking {len(files)} file(s)...")
        
        all_valid = True
        for filepath in sorted(files):
            if not self.check_file(Path(filepath)):
                all_valid = False
        
        # Report results
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} error(s):", file=sys.stderr)
            for error in self.errors:
                print(f"  {error}", file=sys.stderr)
        
        if self.warnings:
            print(f"\n⚠️  Found {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if all_valid and not self.errors:
            print("\n✅ All CLI snippets are valid!")
            return 0
        else:
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate CLI examples in documentation files"
    )
    parser.add_argument(
        '--docs',
        default='docs/PHASE2*.md',
        help='Glob pattern for documentation files to check (default: docs/PHASE2*.md)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Find repository root
    root_dir = Path(__file__).parent.parent
    
    checker = SnippetChecker(root_dir, verbose=args.verbose)
    exit_code = checker.check_docs(args.docs)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
