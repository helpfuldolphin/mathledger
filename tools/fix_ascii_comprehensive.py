#!/usr/bin/env python3
"""
Comprehensive ASCII hygiene fixer script for documentation and scripts.

This script fixes ALL non-ASCII characters by either replacing them with
ASCII equivalents or removing them entirely.
"""

import sys
import os
import re
import unicodedata
from pathlib import Path


def fix_ascii_content(content: str) -> str:
    """Fix ALL non-ASCII characters in text content."""
    # Remove BOM (Byte Order Mark) if present
    if content.startswith('\ufeff'):
        content = content[1:]

    # First, handle specific problematic characters that have good ASCII equivalents
    replacements = {
        # Smart quotes
        '"': '"', '"': '"', ''': "'", ''': "'",

        # Dashes
        '—': '--', '–': '-',

        # Box drawing characters
        '┌': '+', '┐': '+', '└': '+', '┘': '+',
        '├': '+', '┤': '+', '┬': '+', '┴': '+', '┼': '+',
        '─': '-', '│': '|',
        '╭': '+', '╮': '+', '╰': '+', '╯': '+',

        # Arrows
        '↓': 'v', '↑': '^', '→': '->', '←': '<-',
        '↗': '/\\', '↘': '\\/', '↖': '/\\', '↙': '\\/',

        # Mathematical symbols
        '≥': '>=', '≤': '<=', '≠': '!=', '×': 'x', '÷': '/',
        '±': '+/-', '≈': '~', '∞': 'inf', '∅': '{}',
        '∈': 'in', '∉': 'not in', '∪': 'U', '∩': '^',
        '⊂': 'subset', '⊃': 'superset', '⊆': 'subset=', '⊇': 'superset=',

        # Greek letters
        'Δ': 'Delta', 'δ': 'delta', 'α': 'alpha', 'β': 'beta',
        'γ': 'gamma', 'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta',
        'θ': 'theta', 'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda',
        'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron',
        'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau',
        'υ': 'upsilon', 'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',

        # Common symbols
        '…': '...', '•': '*', '★': '*', '☆': '*', '◆': '*', '◇': '*',
        '●': '*', '○': 'o', '■': '*', '□': '[]', '▲': '^', '△': '^',
        '▼': 'v', '▽': 'v',
    }

    # Apply specific replacements
    for unicode_char, ascii_replacement in replacements.items():
        content = content.replace(unicode_char, ascii_replacement)

    # For any remaining non-ASCII characters, use Unicode normalization
    # This will convert accented characters to their base forms
    content = unicodedata.normalize('NFD', content)

    # Remove any remaining non-ASCII characters
    content = ''.join(char for char in content if ord(char) < 128)

    return content


def fix_file(file_path: Path) -> bool:
    """Fix ASCII issues in a single file. Returns True if changes were made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        fixed_content = fix_ascii_content(original_content)

        if original_content != fixed_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def find_files_to_fix(directories: list[str]) -> list[Path]:
    """Find all files that need ASCII fixes in the specified directories."""
    files_to_fix = []

    # File extensions to check
    check_extensions = {'.md', '.py', '.ps1', '.sh', '.yaml', '.yml', '.json', '.txt', '.sql', '.lean'}

    for directory in directories:
        if not os.path.exists(directory):
            continue

        for file_path in Path(directory).rglob('*'):
            if (file_path.is_file() and
                file_path.suffix.lower() in check_extensions):
                files_to_fix.append(file_path)

    return files_to_fix


def main():
    """Main function to fix ASCII issues."""
    # Directories to check for ASCII fixes
    check_dirs = ['docs/', 'scripts/']

    # Files that are allowed to have non-ASCII content (mathematical symbols, etc.)
    allowed_non_ascii_files = {
        'docs/API_REFERENCE.md',  # Contains mathematical symbols
        'docs/whitepaper.md',     # Contains mathematical symbols
        'docs/theory_packs.md',   # Contains mathematical symbols
        'docs/edge_setup.md',     # Contains mathematical symbols
        'docs/M2_WIRING_STATUS.md', # Contains mathematical symbols
        'docs/perf/modus_ponens_indexing.md', # Contains mathematical symbols
        'docs/progress.md',       # Contains status symbols
    }

    # Find files to fix
    files_to_fix = find_files_to_fix(check_dirs)

    # Filter out allowed files (normalize paths for comparison)
    files_to_process = []
    for f in files_to_fix:
        # Convert to forward slashes for consistent comparison
        normalized_path = str(f).replace('\\', '/')
        if normalized_path not in allowed_non_ascii_files:
            files_to_process.append(f)

    # Process files
    fixed_count = 0
    for file_path in files_to_process:
        if fix_file(file_path):
            print(f"Fixed: {file_path}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")
    return 0


if __name__ == '__main__':
    sys.exit(main())
