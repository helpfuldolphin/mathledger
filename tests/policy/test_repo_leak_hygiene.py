"""
Repository Leak Hygiene Tripwire Tests

Ensures founder-private negotiation artifacts cannot leak into version control.
Scans tracked files for banned negotiation terms that should never appear in
public-facing repository content.

SCOPE:
- Scans all files that would be committed (respects .gitignore)
- Allowlist: docs/internal/** (if gitignored), _founder_notes/** (if gitignored)
- Fails if banned terms appear in tracked files

BANNED TERMS (negotiation-sensitive):
- valuation, term sheet, strike price, discount (in negotiation context)
- earnout, exclusivity, M&A, acquisition price
- Specific dollar amounts ($X, $XM, million in financial context)

ALLOWLIST PATHS (must be gitignored):
- _founder_notes/**
- docs/internal/**
- _private/**
- _negotiation/**

SHADOW MODE — observational tripwire.
"""

import re
import subprocess
from pathlib import Path
from typing import List, NamedTuple, Set

import pytest


# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

# Banned terms that should never appear in tracked repository files
# These indicate negotiation-sensitive content that must stay private
#
# NOTE: Patterns are designed to minimize false positives:
# - "valuation" in logic context (truth valuation) is allowed
# - LaTeX math ($...$) is not flagged as dollar amounts
# - "mutual exclusivity" in code is allowed
# - "million ops/sec" technical context is allowed
BANNED_NEGOTIATION_TERMS: List[str] = [
    # Financial/valuation terms (business context, not logic)
    r"\bcompany\s+valuation\b",
    r"\bstartup\s+valuation\b",
    r"\bpre-money\s+valuation\b",
    r"\bpost-money\s+valuation\b",
    r"\bvaluation\s+of\s+\$",  # "valuation of $X"
    r"\bvaluation:\s*\$",      # "Valuation: $X"
    r"\bterm\s+sheet\b",
    r"\bstrike\s+price\b",
    r"\bearnout\b",
    r"\bM&A\b",
    r"\bacquisition\s+price\b",
    r"\bacquisition\s+target\b",
    # Dollar amounts in business context (not LaTeX math)
    # Match $XM, $XK, $X million but NOT $\beta$ or $X$ math
    r"\$\d+\.?\d*[KMB]\b",           # $1M, $500K, $2.5B
    r"\$\d+\.?\d*\s*million\b",      # "$50 million"
    r"\$\d+\.?\d*\s*billion\b",      # "$2 billion"
    # Discount in negotiation context only
    r"\bdiscount\s+removal\b",
    r"\bdiscount[\s-]+adjusted\s+value\b",
    r"\bdiscount[\s-]+adjusted\s+valuation\b",
    # Negotiation-specific phrases
    r"\bnegotiation\s+position\b",
    r"\bwalk[\s-]?away\s+condition\b",
    r"\bwalk[\s-]?away\s+price\b",
    r"\bleverage\s+execution\b",
    r"\bmaximize\s+valuation\b",
    # Acquirer in business context
    r"\bthe\s+acquirer\b",
    r"\bpotential\s+acquirer\b",
    r"\bacquirer\s+owns\b",
]

# Paths that are allowlisted (should be gitignored, but if scanned, don't fail)
ALLOWLIST_PATH_PATTERNS: List[str] = [
    "_founder_notes/",
    "docs/internal/",
    "_private/",
    "_negotiation/",
    "_internal_strategy/",
    # Test files themselves (they contain banned terms as documentation)
    "tests/policy/test_repo_leak_hygiene.py",
    # Policy documentation that defines the banned terms
    "docs/system_law/DOC_PUBLICATION_BOUNDARY.md",
]

# File extensions to scan
SCANNABLE_EXTENSIONS: Set[str] = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".rst", ".tex", ".html", ".css", ".js", ".ts",
}

# Files to always skip (binary, generated, etc.)
SKIP_FILES: Set[str] = {
    "uv.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
}


# =============================================================================
# Types
# =============================================================================

class LeakViolation(NamedTuple):
    """A detected leak-risk violation."""
    file_path: str
    line_num: int
    term: str
    context: str  # Line excerpt


# =============================================================================
# Scanning Functions
# =============================================================================

def get_tracked_files() -> List[Path]:
    """
    Get list of files tracked by git (respects .gitignore).

    Uses `git ls-files` to get only tracked files.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [REPO_ROOT / f for f in result.stdout.strip().split("\n") if f]
        return [f for f in files if f.exists() and f.is_file()]
    except subprocess.CalledProcessError:
        # Fallback: scan all files (less accurate)
        return list(REPO_ROOT.rglob("*"))


def is_allowlisted(file_path: Path) -> bool:
    """Check if file is in an allowlisted path."""
    rel_path = str(file_path.relative_to(REPO_ROOT)).replace("\\", "/")
    for pattern in ALLOWLIST_PATH_PATTERNS:
        if rel_path.startswith(pattern) or pattern in rel_path:
            return True
    return False


def is_scannable(file_path: Path) -> bool:
    """Check if file should be scanned."""
    if file_path.name in SKIP_FILES:
        return False
    if file_path.suffix.lower() not in SCANNABLE_EXTENSIONS:
        return False
    if is_allowlisted(file_path):
        return False
    return True


def scan_file_for_violations(file_path: Path) -> List[LeakViolation]:
    """Scan a single file for banned negotiation terms."""
    violations: List[LeakViolation] = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return violations

    rel_path = str(file_path.relative_to(REPO_ROOT))

    for line_num, line in enumerate(content.splitlines(), start=1):
        line_lower = line.lower()
        for term_pattern in BANNED_NEGOTIATION_TERMS:
            if re.search(term_pattern, line_lower, re.IGNORECASE):
                # Extract the matched term for reporting
                match = re.search(term_pattern, line_lower, re.IGNORECASE)
                if match:
                    violations.append(LeakViolation(
                        file_path=rel_path,
                        line_num=line_num,
                        term=match.group(),
                        context=line.strip()[:100],
                    ))

    return violations


def scan_repository() -> List[LeakViolation]:
    """Scan entire repository for leak-risk violations."""
    all_violations: List[LeakViolation] = []

    tracked_files = get_tracked_files()

    for file_path in tracked_files:
        if is_scannable(file_path):
            violations = scan_file_for_violations(file_path)
            all_violations.extend(violations)

    return all_violations


def format_violations_report(violations: List[LeakViolation]) -> str:
    """Format violations into a readable report."""
    if not violations:
        return "No leak-risk violations found."

    lines = [
        "LEAK-RISK VIOLATIONS DETECTED",
        "=" * 60,
        "",
        f"Total violations: {len(violations)}",
        "",
        "Violations by file:",
        "",
    ]

    # Group by file
    by_file: dict = {}
    for v in violations:
        if v.file_path not in by_file:
            by_file[v.file_path] = []
        by_file[v.file_path].append(v)

    for file_path, file_violations in sorted(by_file.items()):
        lines.append(f"  {file_path}:")
        for v in file_violations:
            lines.append(f"    L{v.line_num}: [{v.term}] \"{v.context[:60]}...\"")
        lines.append("")

    lines.extend([
        "REMEDIATION:",
        "  1. Move sensitive content to _founder_notes/ or docs/internal/",
        "  2. Ensure those directories are in .gitignore",
        "  3. Use neutral language in public-facing docs",
        "  4. See docs/system_law/DOC_PUBLICATION_BOUNDARY.md for policy",
    ])

    return "\n".join(lines)


# =============================================================================
# Tripwire Tests
# =============================================================================

class TestRepoLeakHygiene:
    """Tripwire tests for repository leak hygiene."""

    def test_no_banned_terms_in_tracked_files(self):
        """
        Tracked files must not contain banned negotiation terms.

        This test scans all git-tracked files for terms that indicate
        negotiation-sensitive content that should remain private.
        """
        violations = scan_repository()

        if violations:
            report = format_violations_report(violations)
            pytest.fail(
                f"Leak-risk violations found in tracked files:\n\n{report}\n\n"
                f"Move sensitive content to gitignored paths (_founder_notes/, docs/internal/)"
            )

    def test_founder_notes_gitignored(self):
        """_founder_notes/ must be in .gitignore."""
        gitignore_path = REPO_ROOT / ".gitignore"
        if not gitignore_path.exists():
            pytest.fail(".gitignore file not found")

        content = gitignore_path.read_text()
        assert "_founder_notes/" in content, (
            "_founder_notes/ is not in .gitignore. "
            "Add it to prevent accidental commit of private documents."
        )

    def test_docs_internal_gitignored(self):
        """docs/internal/ must be in .gitignore."""
        gitignore_path = REPO_ROOT / ".gitignore"
        if not gitignore_path.exists():
            pytest.fail(".gitignore file not found")

        content = gitignore_path.read_text()
        assert "docs/internal/" in content, (
            "docs/internal/ is not in .gitignore. "
            "Add it to prevent accidental commit of internal review documents."
        )

    def test_private_dirs_gitignored(self):
        """All private directories must be in .gitignore."""
        gitignore_path = REPO_ROOT / ".gitignore"
        if not gitignore_path.exists():
            pytest.fail(".gitignore file not found")

        content = gitignore_path.read_text()

        required_ignores = [
            "_founder_notes/",
            "_private/",
            "_negotiation/",
            "docs/internal/",
        ]

        missing = [d for d in required_ignores if d not in content]
        assert not missing, (
            f"Private directories not in .gitignore: {missing}\n"
            f"Add them to prevent accidental commit."
        )


class TestBannedTermDetection:
    """Verify the banned term detection works correctly."""

    @pytest.mark.parametrize("term,text,should_match", [
        # Business valuation terms - SHOULD match
        ("company valuation", "The company valuation is $10M", True),
        ("valuation: $", "Valuation: $50M-$200M", True),
        # Logic valuation - should NOT match
        ("valuation", "truth valuation {q=T}", False),
        ("valuation", "Since valuation makes the formula false", False),
        # Term sheet - SHOULD match
        ("term sheet", "Sign the term sheet", True),
        # Dollar amounts with suffixes - SHOULD match
        ("$1M", "Worth $1M", True),
        ("$500K", "Budget: $500K", True),
        ("$50M", "Valuation: $50M-$200M", True),
        # LaTeX math dollars - should NOT match
        ("$\\beta$", "Does $\\beta$ remain stable", False),
        ("$128", "| $128 \\pm 0.5$ |", False),
        # Million in technical context - should NOT match
        ("million ops", "SHA-256: ~1-2 million ops/sec", False),
        # Earnout - SHOULD match
        ("earnout", "The earnout clause", True),
        # M&A - SHOULD match
        ("M&A", "M&A transaction", True),
        # Discount removal - SHOULD match
        ("discount removal", "discount removal evidence", True),
        # Plain discount - should NOT match
        ("discount", "discount factor in math", False),
        # Mutual exclusivity (code) - should NOT match
        ("exclusivity", "Validate mutual exclusivity", False),
        # The acquirer - SHOULD match
        ("the acquirer", "The acquirer owns the substrate", True),
        # Acquirer in other context - should NOT match
        ("acquirer", "data acquirer module", False),
    ])
    def test_term_detection(self, term: str, text: str, should_match: bool):
        """Verify banned terms are detected correctly."""
        text_lower = text.lower()
        matched = False

        for pattern in BANNED_NEGOTIATION_TERMS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matched = True
                break

        if should_match:
            assert matched, f"Expected to match '{term}' in '{text}'"
        else:
            assert not matched, f"Should NOT match '{term}' in '{text}' but did"


class TestAllowlistPaths:
    """Verify allowlist paths are correctly identified."""

    @pytest.mark.parametrize("path,expected_allowlisted", [
        ("_founder_notes/execution/plan.md", True),
        ("docs/internal/reviews/review.md", True),
        ("_private/notes.md", True),
        ("docs/system_law/spec.md", False),
        ("backend/health/adapter.py", False),
        ("tests/policy/test_repo_leak_hygiene.py", True),  # Self-allowlisted
        ("docs/system_law/DOC_PUBLICATION_BOUNDARY.md", True),  # Policy doc allowlisted
    ])
    def test_allowlist_detection(self, path: str, expected_allowlisted: bool):
        """Verify allowlist path detection."""
        file_path = REPO_ROOT / path
        result = is_allowlisted(file_path)
        assert result == expected_allowlisted, (
            f"Path '{path}' allowlist status: expected {expected_allowlisted}, got {result}"
        )
