#!/usr/bin/env python3
"""
Ultimate ASCII Validator - The Death Note of Non-ASCII Characters
================================================================

This validator banishes all non-ASCII glyphs except whitelisted mathematical symbols.
Like a Shinigami with a broom, it purges the repository of all impurities.

Anime Energy:
- Death Note: Every violation is a name in the notebook — erased before it pollutes
- Naruto: Perfect chakra control keeps the code stable
- Dragon Ball Z: SSJ power obliterates stray non-ASCII with Final Flash
- One Piece: Three swords of pre-commit discipline cut through conflicts
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Set, Dict, Tuple


class ASCIIValidator:
    """The ultimate ASCII validator - purger of all impurities."""

    def __init__(self):
        # Whitelisted mathematical symbols (allowed in technical docs)
        self.math_symbols = {
            # Logical operators
            '∧', '∨', '¬', '→', '↔', '∀', '∃', '∈', '∉', '⊂', '⊃', '⊆', '⊇',
            '∪', '∩', '∅', '∞', '±', '×', '÷', '√', '∑', '∏', '∫', '∂', '∆',
            # Greek letters (common in math)
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'λ', 'μ', 'π', 'σ', 'τ',
            'φ', 'χ', 'ψ', 'ω', 'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Λ',
            'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
            # Comparison operators
            '≤', '≥', '≠', '≈', '≡', '≢', '≺', '≻', '≼', '≽', '≪', '≫',
            # Set theory
            '∅', 'ℕ', 'ℤ', 'ℚ', 'ℝ', 'ℂ', 'ℙ', 'ℵ',
            # Arrows
            '←', '↑', '↓', '↖', '↗', '↘', '↙', '↩', '↪', '↶', '↷', '↺', '↻',
            # Other mathematical symbols
            '∠', '⊥', '∥', '⌊', '⌋', '⌈', '⌉', '⌊', '⌋', '⌈', '⌉',
            '⟨', '⟩', '⟦', '⟧', '⟪', '⟫', '⟬', '⟭', '⟮', '⟯',
            '⟰', '⟱', '⟲', '⟳', '⟴', '⟵', '⟶', '⟷', '⟸', '⟹', '⟺',
            '⟻', '⟼', '⟽', '⟾', '⟿', '⤀', '⤁', '⤂', '⤃', '⤄', '⤅', '⤆',
            '⤇', '⤈', '⤉', '⤊', '⤋', '⤌', '⤍', '⤎', '⤏', '⤐', '⤑', '⤒',
            '⤓', '⤔', '⤕', '⤖', '⤗', '⤘', '⤙', '⤚', '⤛', '⤜', '⤝', '⤞',
            '⤟', '⤠', '⤡', '⤢', '⤣', '⤤', '⤥', '⤦', '⤧', '⤨', '⤩', '⤪',
            '⤫', '⤬', '⤭', '⤮', '⤯', '⤰', '⤱', '⤲', '⤳', '⤴', '⤵', '⤶',
            '⤷', '⤸', '⤹', '⤺', '⤻', '⤼', '⤽', '⤾', '⤿', '⥀', '⥁', '⥂',
            '⥃', '⥄', '⥅', '⥆', '⥇', '⥈', '⥉', '⥊', '⥋', '⥌', '⥍', '⥎',
            '⥏', '⥐', '⥑', '⥒', '⥓', '⥔', '⥕', '⥖', '⥗', '⥘', '⥙', '⥚',
            '⥛', '⥜', '⥝', '⥞', '⥟', '⥠', '⥡', '⥢', '⥣', '⥤', '⥥', '⥦',
            '⥧', '⥨', '⥩', '⥪', '⥫', '⥬', '⥭', '⥮', '⥯', '⥰', '⥱', '⥲',
            '⥳', '⥴', '⥵', '⥶', '⥷', '⥸', '⥹', '⥺', '⥻', '⥼', '⥽', '⥾',
            '⥿'
        }

        # Critical non-ASCII characters that must be eliminated
        self.forbidden_chars = {
            # Smart quotes and punctuation
            '"', '"', ''', ''', '—', '–', '…', '•', '◦', '▪', '▫', '‣', '⁃',
            # Common emojis and symbols
            '✅', '❌', '⚠️', 'ℹ️', '🔍', '📊', '📈', '📉', '🎯', '🚀', '💡',
            '🔧', '📝', '📋', '🔗', '⭐', '🔥', '💯', '🎉', '🎊', '🎁', '🎂',
            '🎈', '🎪', '🎭', '🎨', '🎬', '🎵', '🎶', '🎸', '🎹', '🎺', '🎻',
            '🎼', '🎽', '🎾', '🎿', '🏀', '🏁', '🏂', '🏃', '🏄', '🏅', '🏆',
            '🏇', '🏈', '🏉', '🏊', '🏋', '🏌', '🏍', '🏎', '🏏', '🏐', '🏑',
            '🏒', '🏓', '🏔', '🏕', '🏖', '🏗', '🏘', '🏙', '🏚', '🏛', '🏜',
            '🏝', '🏞', '🏟', '🏠', '🏡', '🏢', '🏣', '🏤', '🏥', '🏦', '🏧',
            '🏨', '🏩', '🏪', '🏫', '🏬', '🏭', '🏮', '🏯', '🏰', '🏱', '🏲',
            '🏳', '🏴', '🏵', '🏶', '🏷', '🏸', '🏹', '🏺'
        }

        # File extensions to check
        self.check_extensions = {'.md', '.py', '.ps1', '.sh', '.yaml', '.yml', '.json', '.txt', '.sql', '.lean'}

        # File extensions to skip
        self.skip_extensions = {'.pdf', '.log', '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.db', '.sqlite', '.svg', '.png', '.jpg', '.jpeg', '.gif', '.ico'}

        # Files that are allowed to have mathematical symbols
        self.math_whitelist = {
            'docs/API_REFERENCE.md',
            'docs/whitepaper.md',
            'docs/theory_packs.md',
            'docs/edge_setup.md',
            'docs/M2_WIRING_STATUS.md',
            'docs/perf/modus_ponens_indexing.md',
            'docs/progress.md'
        }

    def is_ascii_with_math(self, content: str, file_path: str) -> Tuple[bool, List[str]]:
        """
        Check if content is ASCII-only with allowed mathematical symbols.
        Returns (is_valid, list_of_violations)
        """
        violations = []

        # Check if file is in math whitelist
        normalized_path = str(file_path).replace('\\', '/')
        is_math_file = normalized_path in self.math_whitelist

        for i, char in enumerate(content):
            if ord(char) > 127:  # Non-ASCII character
                if is_math_file and char in self.math_symbols:
                    continue  # Allow mathematical symbols in whitelisted files
                elif char in self.forbidden_chars:
                    violations.append(f"Position {i}: Forbidden character '{char}' (U+{ord(char):04X})")
                else:
                    violations.append(f"Position {i}: Non-ASCII character '{char}' (U+{ord(char):04X})")

        return len(violations) == 0, violations

    def scan_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Scan a single file for ASCII violations."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.is_ascii_with_math(content, str(file_path))
        except UnicodeDecodeError:
            return False, [f"Unicode decode error - file may not be UTF-8"]
        except Exception as e:
            return False, [f"Error reading file: {e}"]

    def scan_directory(self, directory: str) -> Dict[str, List[str]]:
        """Scan directory for ASCII violations."""
        violations = {}

        if not os.path.exists(directory):
            return violations

        for file_path in Path(directory).rglob('*'):
            if (file_path.is_file() and
                file_path.suffix.lower() not in self.skip_extensions and
                file_path.suffix.lower() in self.check_extensions):

                is_valid, file_violations = self.scan_file(file_path)
                if not is_valid:
                    violations[str(file_path)] = file_violations

        return violations

    def generate_ascii_art_report(self, violations: Dict[str, List[str]]) -> str:
        """Generate ASCII art report with anime energy."""
        if not violations:
            return """
+==============================================================================+
|                          ASCII VALIDATION SUCCESS                            |
|                                                                              |
|  No impurities detected! The repository shines like a polished katana!      |
|                                                                              |
|  Like a Shinigami with perfect chakra control, all files are pure!          |
|  The code flows like a perfect jutsu - stable and powerful!                 |
|                                                                              |
|  FINAL FLASH: All non-ASCII characters have been obliterated!               |
+==============================================================================+
"""
        else:
            report = """
+==============================================================================+
|                        ASCII VIOLATIONS DETECTED                             |
|                                                                              |
|  The Death Note has been written! These files must be purified:             |
|                                                                              |
"""
            for file_path, file_violations in violations.items():
                report += f"|  {file_path:<70} |\n"
                for violation in file_violations[:3]:  # Show first 3 violations
                    report += f"|     X {violation:<65} |\n"
                if len(file_violations) > 3:
                    report += f"|     ... and {len(file_violations) - 3} more violations{' ' * 45} |\n"
                report += "|                                                                              |\n"

            report += """|                                                                              |
|  Use the Final Flash to obliterate these impurities!                       |
|  Three swords of pre-commit discipline will cut through conflicts!         |
+==============================================================================+
"""
            return report

    def validate_repository(self, directories: List[str] = None) -> bool:
        """Validate entire repository for ASCII compliance."""
        if directories is None:
            directories = ['docs/', 'scripts/']

        print("Scanning repository with the power of a Super Saiyan...")
        print("Three Sword Style: Cutting through all impurities...")
        print()

        all_violations = {}
        for directory in directories:
            print(f"Scanning {directory}...")
            violations = self.scan_directory(directory)
            all_violations.update(violations)

        print()
        print(self.generate_ascii_art_report(all_violations))

        if all_violations:
            print("\nTo fix these violations:")
            print("   1. Run: python tools/fix_ascii.py")
            print("   2. Or manually replace non-ASCII characters with ASCII equivalents")
            print("   3. Re-run this validator to confirm fixes")
            return False
        else:
            print("\nRepository is now pure! All non-ASCII impurities have been banished!")
            return True


def main():
    """Main function - the ultimate ASCII validator."""
    print("Cursor C - Hygiene Marshal ASCII Validator")
    print("Death Note: Every violation is a name in the notebook")
    print("Final Flash: Obliterating all non-ASCII impurities!")
    print()

    validator = ASCIIValidator()
    success = validator.validate_repository()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
