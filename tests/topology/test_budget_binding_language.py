"""
Language regression guard for budget_binding.py

Ensures the module does not contain forbidden phrases from the canonical
language constraints list.

This is a codebase regression guard only — non-gating in runtime.

SHADOW MODE — observational only.
"""

from pathlib import Path

import pytest

from backend.governance.language_constraints import (
    FORBIDDEN_PHRASES,
    scan_text_for_violations,
)


ROOT = Path(__file__).resolve().parents[2]
TARGET_FILE = ROOT / "backend" / "topology" / "first_light" / "budget_binding.py"


def test_budget_binding_no_forbidden_phrases():
    """budget_binding.py must not contain forbidden language phrases."""
    if not TARGET_FILE.exists():
        pytest.skip(f"Target file not found: {TARGET_FILE}")

    content = TARGET_FILE.read_text(encoding="utf-8")
    violations = scan_text_for_violations(content)

    if violations:
        msg_lines = [
            "",
            f"Forbidden phrases found in {TARGET_FILE.name}:",
            "",
        ]
        for line_num, phrase, context in violations:
            msg_lines.append(f"  Line {line_num}: \"{phrase}\"")
            msg_lines.append(f"    Context: {context}")
            msg_lines.append("")
        msg_lines.append("See: backend/governance/language_constraints.py")
        msg_lines.append("Fix: Replace with neutral equivalent (e.g., 'criteria met', 'check completed')")
        pytest.fail("\n".join(msg_lines))


def test_forbidden_phrases_coverage():
    """Verify test covers all canonical forbidden phrases."""
    # Sanity check: ensure we're testing against the full canonical list
    assert len(FORBIDDEN_PHRASES) >= 9, (
        f"Expected at least 9 forbidden phrases, got {len(FORBIDDEN_PHRASES)}. "
        "Has the canonical list been truncated?"
    )
    assert "calibration passed" in FORBIDDEN_PHRASES, (
        "'calibration passed' must be in forbidden list"
    )
