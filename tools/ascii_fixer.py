#!/usr/bin/env python3
"""
Ultimate ASCII Fixer - The Final Flash of Character Purity
=========================================================

This tool obliterates all non-ASCII characters with the power of a Super Saiyan.
Like Goku's Final Flash, it purges the repository of all impurities in one blast!

Anime Energy:
- Dragon Ball Z: SSJ power obliterates stray non-ASCII with Final Flash
- Death Note: Every violation is erased from the notebook
- Naruto: Perfect chakra control transforms impurities into pure ASCII
- One Piece: Three swords cut through all character conflicts
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


class ASCIIFixer:
    """The ultimate ASCII fixer - purger of all impurities."""

    def __init__(self):
        # Comprehensive Unicode to ASCII replacement mapping
        self.replacements = {
            # Mathematical symbols (convert to ASCII equivalents)
            '≥': '>=', '≤': '<=', '≠': '!=', '≈': '~=', '∞': 'inf',
            '±': '+/-', '×': 'x', '÷': '/', '√': 'sqrt', '∑': 'sum',
            '∏': 'prod', '∫': 'int', '∂': 'd', '∆': 'delta',

            # Greek letters (convert to ASCII names)
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
            'λ': 'lambda', 'μ': 'mu', 'π': 'pi', 'σ': 'sigma',
            'τ': 'tau', 'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
            'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta',
            'Ε': 'Epsilon', 'Ζ': 'Zeta', 'Η': 'Eta', 'Θ': 'Theta',
            'Λ': 'Lambda', 'Μ': 'Mu', 'Ν': 'Nu', 'Ξ': 'Xi',
            'Ο': 'Omicron', 'Π': 'Pi', 'Ρ': 'Rho', 'Σ': 'Sigma',
            'Τ': 'Tau', 'Υ': 'Upsilon', 'Φ': 'Phi', 'Χ': 'Chi',
            'Ψ': 'Psi', 'Ω': 'Omega',

            # Logical operators
            '∧': 'AND', '∨': 'OR', '¬': 'NOT', '→': '->', '↔': '<->',
            '∀': 'FORALL', '∃': 'EXISTS', '∈': 'in', '∉': 'not in',
            '⊂': 'subset', '⊃': 'superset', '⊆': 'subseteq', '⊇': 'supseteq',
            '∪': 'union', '∩': 'intersection', '∅': 'empty',

            # Comparison operators
            '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~=', '≡': '==',
            '≢': '!=', '≺': '<', '≻': '>', '≼': '<=', '≽': '>=',
            '≪': '<<', '≫': '>>',

            # Punctuation and symbols
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '--', '…': '...', '•': '*', '◦': 'o',
            '▪': '[', '▫': ']', '‣': '>', '⁃': '-',

            # Arrows
            '←': '<-', '↑': '^', '↓': 'v', '↖': '<^', '↗': '^>',
            '↘': 'v>', '↙': '<v', '↩': '<-', '↪': '->', '↶': '<^',
            '↷': '^>', '↺': '<-', '↻': '->',

            # Status symbols and emojis
            '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARNING]', 'ℹ️': '[INFO]',
            '🔍': '[SEARCH]', '📊': '[CHART]', '📈': '[UP]', '📉': '[DOWN]',
            '🎯': '[TARGET]', '🚀': '[ROCKET]', '💡': '[IDEA]', '🔧': '[TOOL]',
            '📝': '[NOTE]', '📋': '[CLIPBOARD]', '🔗': '[LINK]', '⭐': '[STAR]',
            '🔥': '[FIRE]', '💯': '[100]', '🎉': '[PARTY]', '🎊': '[CONFETTI]',
            '🎁': '[GIFT]', '🎂': '[CAKE]', '🎈': '[BALLOON]', '🎪': '[CIRCUS]',
            '🎭': '[THEATER]', '🎨': '[ART]', '🎬': '[MOVIE]', '🎵': '[MUSIC]',
            '🎶': '[NOTES]', '🎸': '[GUITAR]', '🎹': '[PIANO]', '🎺': '[TRUMPET]',
            '🎻': '[VIOLIN]', '🎼': '[SCORE]', '🎽': '[RUNNING]', '🎾': '[TENNIS]',
            '🎿': '[SKIING]', '🏀': '[BASKETBALL]', '🏁': '[FINISH]', '🏂': '[SNOWBOARD]',
            '🏃': '[RUNNING]', '🏄': '[SURFING]', '🏅': '[MEDAL]', '🏆': '[TROPHY]',
            '🏇': '[HORSE]', '🏈': '[FOOTBALL]', '🏉': '[RUGBY]', '🏊': '[SWIMMING]',
            '🏋': '[WEIGHT]', '🏌': '[GOLF]', '🏍': '[MOTORCYCLE]', '🏎': '[RACE CAR]',
            '🏏': '[CRICKET]', '🏐': '[VOLLEYBALL]', '🏑': '[HOCKEY]', '🏒': '[HOCKEY]',
            '🏓': '[PING PONG]', '🏔': '[MOUNTAIN]', '🏕': '[CAMPING]', '🏖': '[BEACH]',
            '🏗': '[CONSTRUCTION]', '🏘': '[HOUSES]', '🏙': '[CITY]', '🏚': '[HOUSE]',
            '🏛': '[BUILDING]', '🏜': '[DESERT]', '🏝': '[ISLAND]', '🏞': '[PARK]',
            '🏟': '[STADIUM]', '🏠': '[HOUSE]', '🏡': '[HOUSE]', '🏢': '[OFFICE]',
            '🏣': '[POST OFFICE]', '🏤': '[EUROPEAN POST OFFICE]', '🏥': '[HOSPITAL]',
            '🏦': '[BANK]', '🏧': '[ATM]', '🏨': '[HOTEL]', '🏩': '[LOVE HOTEL]',
            '🏪': '[STORE]', '🏫': '[SCHOOL]', '🏬': '[DEPARTMENT STORE]', '🏭': '[FACTORY]',
            '🏮': '[LANTERN]', '🏯': '[CASTLE]', '🏰': '[CASTLE]', '🏱': '[JAPANESE POST OFFICE]',
            '🏲': '[JAPANESE BANK]', '🏳': '[FLAG]', '🏴': '[FLAG]', '🏵': '[ROSETTE]',
            '🏶': '[LABEL]', '🏷': '[LABEL]', '🏸': '[BADMINTON]', '🏹': '[BOW AND ARROW]',
            '🏺': '[AMPHORA]',

            # Skin tone modifiers (remove)
            '🏻': '', '🏼': '', '🏽': '', '🏾': '', '🏿': '',

            # Additional mathematical symbols
            '∠': 'angle', '⊥': 'perp', '∥': 'parallel', '⌊': 'floor', '⌋': 'floor',
            '⌈': 'ceil', '⌉': 'ceil', '⟨': '<', '⟩': '>', '⟦': '[', '⟧': ']',
            '⟪': '[', '⟫': ']', '⟬': '[', '⟭': ']', '⟮': '[', '⟯': ']',
            '⟰': 'up', '⟱': 'down', '⟲': 'left', '⟳': 'right', '⟴': 'up',
            '⟵': '<-', '⟶': '->', '⟷': '<->', '⟸': '<=', '⟹': '=>', '⟺': '<=>',
            '⟻': '<-', '⟼': '->', '⟽': '<=', '⟾': '=>', '⟿': '<=>',
            '⤀': 'up', '⤁': 'down', '⤂': 'left', '⤃': 'right', '⤄': 'up',
            '⤅': 'down', '⤆': 'left', '⤇': 'right', '⤈': 'up', '⤉': 'down',
            '⤊': 'left', '⤋': 'right', '⤌': 'up', '⤍': 'down', '⤎': 'left',
            '⤏': 'right', '⤐': 'up', '⤑': 'down', '⤒': 'left', '⤓': 'right',
            '⤔': 'up', '⤕': 'down', '⤖': 'left', '⤗': 'right', '⤘': 'up',
            '⤙': 'down', '⤚': 'left', '⤛': 'right', '⤜': 'up', '⤝': 'down',
            '⤞': 'left', '⤟': 'right', '⤠': 'up', '⤡': 'down', '⤢': 'left',
            '⤣': 'right', '⤤': 'up', '⤥': 'down', '⤦': 'left', '⤧': 'right',
            '⤨': 'up', '⤩': 'down', '⤪': 'left', '⤫': 'right', '⤬': 'up',
            '⤭': 'down', '⤮': 'left', '⤯': 'right', '⤰': 'up', '⤱': 'down',
            '⤲': 'left', '⤳': 'right', '⤴': 'up', '⤵': 'down', '⤶': 'left',
            '⤷': 'right', '⤸': 'up', '⤹': 'down', '⤺': 'left', '⤻': 'right',
            '⤼': 'up', '⤽': 'down', '⤾': 'left', '⤿': 'right', '⥀': 'up',
            '⥁': 'down', '⥂': 'left', '⥃': 'right', '⥄': 'up', '⥅': 'down',
            '⥆': 'left', '⥇': 'right', '⥈': 'up', '⥉': 'down', '⥊': 'left',
            '⥋': 'right', '⥌': 'up', '⥍': 'down', '⥎': 'left', '⥏': 'right',
            '⥐': 'up', '⥑': 'down', '⥒': 'left', '⥓': 'right', '⥔': 'up',
            '⥕': 'down', '⥖': 'left', '⥗': 'right', '⥘': 'up', '⥙': 'down',
            '⥚': 'left', '⥛': 'right', '⥜': 'up', '⥝': 'down', '⥞': 'left',
            '⥟': 'right', '⥠': 'up', '⥡': 'down', '⥢': 'left', '⥣': 'right',
            '⥤': 'up', '⥥': 'down', '⥦': 'left', '⥧': 'right', '⥨': 'up',
            '⥩': 'down', '⥪': 'left', '⥫': 'right', '⥬': 'up', '⥭': 'down',
            '⥮': 'left', '⥯': 'right', '⥰': 'up', '⥱': 'down', '⥲': 'left',
            '⥳': 'right', '⥴': 'up', '⥵': 'down', '⥶': 'left', '⥷': 'right',
            '⥸': 'up', '⥹': 'down', '⥺': 'left', '⥻': 'right', '⥼': 'up',
            '⥽': 'down', '⥾': 'left', '⥿': 'right'
        }

        # File extensions to process
        self.process_extensions = {'.md', '.py', '.ps1', '.sh', '.yaml', '.yml', '.json', '.txt', '.sql', '.lean'}

        # File extensions to skip
        self.skip_extensions = {'.pdf', '.log', '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.db', '.sqlite', '.svg', '.png', '.jpg', '.jpeg', '.gif', '.ico'}

        # Files that are allowed to have mathematical symbols (don't fix these)
        self.math_whitelist = {
            'docs/API_REFERENCE.md',
            'docs/whitepaper.md',
            'docs/theory_packs.md',
            'docs/edge_setup.md',
            'docs/M2_WIRING_STATUS.md',
            'docs/perf/modus_ponens_indexing.md'
        }

    def fix_content(self, content: str, file_path: str) -> Tuple[str, int]:
        """
        Fix non-ASCII characters in content.
        Returns (fixed_content, number_of_replacements)
        """
        # Check if file is in math whitelist
        normalized_path = str(file_path).replace('\\', '/')
        is_math_file = normalized_path in self.math_whitelist

        fixed_content = content
        replacements = 0

        for unicode_char, ascii_replacement in self.replacements.items():
            if unicode_char in fixed_content:
                # For math files, only replace forbidden characters, not mathematical symbols
                if is_math_file and unicode_char in ['∧', '∨', '¬', '→', '↔', '∀', '∃', '∈', '∉', '⊂', '⊃', '⊆', '⊇', '∪', '∩', '∅', '∞', '±', '×', '÷', '√', '∑', '∏', '∫', '∂', '∆', '≤', '≥', '≠', '≈', '≡', '≢', '≺', '≻', '≼', '≽', '≪', '≫']:
                    continue

                count = fixed_content.count(unicode_char)
                fixed_content = fixed_content.replace(unicode_char, ascii_replacement)
                replacements += count

        return fixed_content, replacements

    def fix_file(self, file_path: Path) -> Tuple[bool, int]:
        """Fix a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            fixed_content, replacements = self.fix_content(content, str(file_path))

            if replacements > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                return True, replacements
            return False, 0

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False, 0

    def fix_directory(self, directory: str) -> Dict[str, int]:
        """Fix all files in directory."""
        fixed_files = {}

        if not os.path.exists(directory):
            return fixed_files

        for file_path in Path(directory).rglob('*'):
            if (file_path.is_file() and
                file_path.suffix.lower() not in self.skip_extensions and
                file_path.suffix.lower() in self.process_extensions):

                was_fixed, replacements = self.fix_file(file_path)
                if was_fixed:
                    fixed_files[str(file_path)] = replacements

        return fixed_files

    def generate_ascii_art_report(self, fixed_files: Dict[str, int]) -> str:
        """Generate ASCII art report."""
        if not fixed_files:
            return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🎌 NO FIXES NEEDED 🎌                                ║
║                                                                              ║
║  ⚔️  The repository is already pure! No impurities detected!                ║
║                                                                              ║
║  Like a perfect jutsu, the code flows with perfect chakra control!          ║
║  The Death Note remains empty - no names to erase!                          ║
║                                                                              ║
║  🔥 FINAL FLASH: Repository is already at maximum power! 🔥                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        else:
            total_replacements = sum(fixed_files.values())
            report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🔥 FINAL FLASH COMPLETE! 🔥                          ║
║                                                                              ║
║  ⚔️  Purified {len(fixed_files)} files with {total_replacements} character replacements! ⚔️  ║
║                                                                              ║
║  🗡️  The Death Note has been written and executed:                         ║
║                                                                              ║
"""
            for file_path, replacements in fixed_files.items():
                report += f"║  📄 {file_path:<50} {replacements:>3} fixes ║\n"

            report += """║                                                                              ║
║  🎉 All non-ASCII impurities have been obliterated! 🎉                    ║
║  ⚔️  Three swords of discipline have cut through all conflicts! ⚔️        ║
║  🔥 The repository now shines like a polished katana! 🔥                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            return report

    def fix_repository(self, directories: List[str] = None) -> bool:
        """Fix entire repository for ASCII compliance."""
        if directories is None:
            directories = ['docs/', 'scripts/']

        print("🔥 Charging up Final Flash with Super Saiyan power...")
        print("⚔️  Three Sword Style: Preparing to cut through all impurities...")
        print("🗡️  Death Note: Writing names of all non-ASCII characters...")
        print()

        all_fixed_files = {}
        for directory in directories:
            print(f"📁 Processing {directory}...")
            fixed_files = self.fix_directory(directory)
            all_fixed_files.update(fixed_files)

        print()
        print(self.generate_ascii_art_report(all_fixed_files))

        if all_fixed_files:
            print("\n🎉 Repository has been purified! All non-ASCII impurities obliterated!")
            print("🔍 Run the validator again to confirm all fixes are complete.")
            return True
        else:
            print("\n✨ Repository was already pure! No fixes needed.")
            return True


def main():
    """Main function - the ultimate ASCII fixer."""
    print("🎌 Cursor C - Hygiene Marshal ASCII Fixer 🎌")
    print("🔥 Final Flash: Obliterating all non-ASCII impurities! 🔥")
    print("⚔️  Three Sword Style: Cutting through all character conflicts! ⚔️")
    print()

    fixer = ASCIIFixer()
    success = fixer.fix_repository()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
