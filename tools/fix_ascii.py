#!/usr/bin/env python3
"""
ASCII hygiene fixer script for documentation and scripts.

This script fixes common non-ASCII characters that cause encoding issues:
- Smart quotes: " " → " "
- Em dashes: — → --
- En dashes: – → -
- Box drawing characters: ┌┐└┘ → +, ─ → -, │ → |
- Other problematic Unicode characters
"""

import sys
import os
from pathlib import Path


def fix_ascii_content(content: str) -> str:
    """Fix common non-ASCII characters in text content."""
    # Remove BOM (Byte Order Mark) if present
    if content.startswith('\ufeff'):
        content = content[1:]

    # Smart quotes
    content = content.replace('"', '"')  # Left double quotation mark
    content = content.replace('"', '"')  # Right double quotation mark
    content = content.replace(''', "'")  # Left single quotation mark
    content = content.replace(''', "'")  # Right single quotation mark

    # Dashes
    content = content.replace('—', '--')  # Em dash
    content = content.replace('–', '-')   # En dash

    # Box drawing characters
    content = content.replace('┌', '+')   # Box drawings light down and right
    content = content.replace('┐', '+')   # Box drawings light down and left
    content = content.replace('└', '+')   # Box drawings light up and right
    content = content.replace('┘', '+')   # Box drawings light up and left
    content = content.replace('├', '+')   # Box drawings light vertical and right
    content = content.replace('┤', '+')   # Box drawings light vertical and left
    content = content.replace('┬', '+')   # Box drawings light down and horizontal
    content = content.replace('┴', '+')   # Box drawings light up and horizontal
    content = content.replace('┼', '+')   # Box drawings light vertical and horizontal
    content = content.replace('─', '-')   # Box drawings light horizontal
    content = content.replace('│', '|')   # Box drawings light vertical
    content = content.replace('╭', '+')   # Box drawings light arc down and right
    content = content.replace('╮', '+')   # Box drawings light arc down and left
    content = content.replace('╰', '+')   # Box drawings light arc up and right
    content = content.replace('╯', '+')   # Box drawings light arc up and left

    # Arrows and symbols
    content = content.replace('↓', 'v')    # Downwards arrow
    content = content.replace('↑', '^')    # Upwards arrow
    content = content.replace('→', '->')   # Rightwards arrow
    content = content.replace('←', '<-')   # Leftwards arrow
    content = content.replace('↗', '/\\')  # North east arrow
    content = content.replace('↘', '\\/') # South east arrow
    content = content.replace('↖', '/\\')  # North west arrow
    content = content.replace('↙', '\\/') # South west arrow

    # Other common problematic characters
    content = content.replace('…', '...')  # Horizontal ellipsis
    content = content.replace('•', '*')    # Bullet
    content = content.replace('≥', '>=')   # Greater-than or equal to
    content = content.replace('≤', '<=')   # Less-than or equal to
    content = content.replace('≠', '!=')   # Not equal to
    content = content.replace('×', 'x')    # Multiplication sign
    content = content.replace('÷', '/')    # Division sign
    content = content.replace('±', '+/-')  # Plus-minus sign
    content = content.replace('≈', '~')    # Almost equal to
    content = content.replace('∞', 'inf')  # Infinity
    content = content.replace('∅', '{}')   # Empty set
    content = content.replace('∈', 'in')   # Element of
    content = content.replace('∉', 'not in') # Not an element of
    content = content.replace('∪', 'U')    # Union
    content = content.replace('∩', '^')    # Intersection
    content = content.replace('⊂', 'subset') # Subset of
    content = content.replace('⊃', 'superset') # Superset of
    content = content.replace('⊆', 'subset=') # Subset of or equal to
    content = content.replace('⊇', 'superset=') # Superset of or equal to

    # Greek letters (common ones)
    content = content.replace('Δ', 'Delta')  # Greek capital letter Delta
    content = content.replace('δ', 'delta')  # Greek small letter delta
    content = content.replace('α', 'alpha')  # Greek small letter alpha
    content = content.replace('β', 'beta')   # Greek small letter beta
    content = content.replace('γ', 'gamma')  # Greek small letter gamma
    content = content.replace('ε', 'epsilon') # Greek small letter epsilon
    content = content.replace('ζ', 'zeta')   # Greek small letter zeta
    content = content.replace('η', 'eta')    # Greek small letter eta
    content = content.replace('θ', 'theta')  # Greek small letter theta
    content = content.replace('ι', 'iota')   # Greek small letter iota
    content = content.replace('κ', 'kappa')  # Greek small letter kappa
    content = content.replace('λ', 'lambda') # Greek small letter lambda
    content = content.replace('μ', 'mu')     # Greek small letter mu
    content = content.replace('ν', 'nu')     # Greek small letter nu
    content = content.replace('ξ', 'xi')     # Greek small letter xi
    content = content.replace('ο', 'omicron') # Greek small letter omicron
    content = content.replace('π', 'pi')     # Greek small letter pi
    content = content.replace('ρ', 'rho')    # Greek small letter rho
    content = content.replace('σ', 'sigma')  # Greek small letter sigma
    content = content.replace('τ', 'tau')    # Greek small letter tau
    content = content.replace('υ', 'upsilon') # Greek small letter upsilon
    content = content.replace('φ', 'phi')    # Greek small letter phi
    content = content.replace('χ', 'chi')    # Greek small letter chi
    content = content.replace('ψ', 'psi')    # Greek small letter psi
    content = content.replace('ω', 'omega')  # Greek small letter omega

    # Emojis and symbols - replace with ASCII equivalents or remove
    content = content.replace('🧠', '[brain]')     # Brain emoji
    content = content.replace('🎯', '[target]')    # Target emoji
    content = content.replace('✅', '[check]')     # Check mark emoji
    content = content.replace('❌', '[x]')         # Cross mark emoji
    content = content.replace('⚠️', '[warning]')   # Warning emoji
    content = content.replace('🔍', '[search]')    # Magnifying glass emoji
    content = content.replace('📊', '[chart]')     # Bar chart emoji
    content = content.replace('📈', '[trend]')     # Trending up emoji
    content = content.replace('📉', '[decline]')   # Trending down emoji
    content = content.replace('🚀', '[rocket]')    # Rocket emoji
    content = content.replace('💡', '[idea]')      # Light bulb emoji
    content = content.replace('🔧', '[tool]')      # Wrench emoji
    content = content.replace('⚡', '[lightning]') # Lightning emoji
    content = content.replace('🔥', '[fire]')      # Fire emoji
    content = content.replace('⭐', '[star]')      # Star emoji
    content = content.replace('🎉', '[celebration]') # Party emoji
    content = content.replace('📝', '[note]')      # Memo emoji
    content = content.replace('🔒', '[lock]')      # Lock emoji
    content = content.replace('🔓', '[unlock]')    # Unlock emoji
    content = content.replace('📋', '[clipboard]') # Clipboard emoji
    content = content.replace('🎨', '[art]')       # Artist palette emoji
    content = content.replace('🏆', '[trophy]')    # Trophy emoji
    content = content.replace('🎪', '[circus]')    # Circus tent emoji
    content = content.replace('🚨', '[alert]')     # Police car light emoji
    content = content.replace('📡', '[satellite]') # Satellite antenna emoji
    content = content.replace('🔔', '[bell]')      # Bell emoji
    content = content.replace('📢', '[megaphone]') # Megaphone emoji
    content = content.replace('📣', '[megaphone]') # Megaphone emoji
    content = content.replace('📯', '[postal]')    # Postal horn emoji
    content = content.replace('📻', '[radio]')     # Radio emoji
    content = content.replace('📱', '[phone]')     # Mobile phone emoji
    content = content.replace('💻', '[computer]')  # Laptop emoji
    content = content.replace('🖥️', '[desktop]')   # Desktop computer emoji
    content = content.replace('⌨️', '[keyboard]')   # Keyboard emoji
    content = content.replace('🖱️', '[mouse]')     # Computer mouse emoji
    content = content.replace('🖨️', '[printer]')   # Printer emoji
    content = content.replace('💾', '[floppy]')    # Floppy disk emoji
    content = content.replace('💿', '[cd]')        # Optical disk emoji
    content = content.replace('📀', '[dvd]')       # DVD emoji
    content = content.replace('🧮', '[abacus]')    # Abacus emoji
    content = content.replace('🎲', '[dice]')      # Game die emoji
    content = content.replace('🎮', '[game]')      # Video game emoji
    content = content.replace('🕹️', '[joystick]')  # Joystick emoji
    content = content.replace('🎯', '[target]')    # Direct hit emoji
    content = content.replace('🎳', '[bowling]')   # Bowling emoji
    content = content.replace('🎴', '[cards]')     # Flower playing cards emoji
    content = content.replace('🃏', '[joker]')     # Joker emoji
    content = content.replace('🀄', '[mahjong]')   # Mahjong red dragon emoji
    content = content.replace('🎰', '[slot]')      # Slot machine emoji
    content = content.replace('🎱', '[8ball]')     # Pool 8 ball emoji
    content = content.replace('🎪', '[circus]')    # Circus tent emoji
    content = content.replace('🎭', '[theater]')   # Performing arts emoji
    content = content.replace('🎨', '[art]')       # Artist palette emoji
    content = content.replace('🎬', '[movie]')     # Clapper board emoji
    content = content.replace('🎤', '[microphone]') # Microphone emoji
    content = content.replace('🎧', '[headphones]') # Headphone emoji
    content = content.replace('🎵', '[note]')      # Musical note emoji
    content = content.replace('🎶', '[notes]')     # Musical notes emoji
    content = content.replace('🎼', '[score]')     # Musical score emoji
    content = content.replace('🎹', '[piano]')     # Musical keyboard emoji
    content = content.replace('🥁', '[drum]')      # Drum emoji
    content = content.replace('🎷', '[saxophone]') # Saxophone emoji
    content = content.replace('🎺', '[trumpet]')   # Trumpet emoji
    content = content.replace('🎻', '[violin]')    # Violin emoji
    content = content.replace('🎸', '[guitar]')    # Guitar emoji
    content = content.replace('🎹', '[piano]')     # Musical keyboard emoji
    content = content.replace('🎤', '[microphone]') # Microphone emoji
    content = content.replace('🎧', '[headphones]') # Headphone emoji
    content = content.replace('🎵', '[note]')      # Musical note emoji
    content = content.replace('🎶', '[notes]')     # Musical notes emoji
    content = content.replace('🎼', '[score]')     # Musical score emoji
    content = content.replace('🎹', '[piano]')     # Musical keyboard emoji
    content = content.replace('🥁', '[drum]')      # Drum emoji
    content = content.replace('🎷', '[saxophone]') # Saxophone emoji
    content = content.replace('🎺', '[trumpet]')   # Trumpet emoji
    content = content.replace('🎻', '[violin]')    # Violin emoji
    content = content.replace('🎸', '[guitar]')    # Guitar emoji

    # Other common Unicode symbols
    content = content.replace('★', '*')      # Black star
    content = content.replace('☆', '*')      # White star
    content = content.replace('◆', '*')      # Black diamond
    content = content.replace('◇', '*')      # White diamond
    content = content.replace('●', '*')      # Black circle
    content = content.replace('○', 'o')      # White circle
    content = content.replace('■', '*')      # Black square
    content = content.replace('□', '[]')     # White square
    content = content.replace('▲', '^')      # Black triangle
    content = content.replace('△', '^')      # White triangle
    content = content.replace('▼', 'v')      # Black triangle down
    content = content.replace('▽', 'v')      # White triangle down

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
