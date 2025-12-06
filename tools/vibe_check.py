#!/usr/bin/env python3
"""
Vibe Check Tool
===============

Enforces the MathLedger Vibe Style Guide on the First Organism path.
Checks for:
1. Banned patterns (TODO, HACK, etc.).
2. Docstring presence in Python files.
3. Professional logging/error message patterns (heuristic).

Usage:
    python tools/vibe_check.py
"""

import os
import re
import sys
from typing import List, Tuple

# Files/Directories to check (The "First Organism Path")
TARGETS = [
    "backend/ledger/ui_events.py",
    "backend/rfl/config.py",
    "backend/rfl/runner.py",
    "backend/lean_interface.py",
    "backend/ledger/ingest.py",
    "attestation/dual_root.py",
    "backend/crypto/dual_root.py", # Check both just in case
    "tests/integration/test_first_organism_dag.py",
    "tests/integration/test_first_organism.py",
    "apps/ui/src/lib/api.ts",
    "backend/orchestrator/parents_routes.py",
    "backend/orchestrator/proof_middleware.py",
    "interface/api/app.py",
    "interface/api/schemas.py",
]

# Banned regex patterns
BANNED_PATTERNS = [
    (r"\bTODO\b", "TODO found (use issue tracker or finish it)"),
    (r"\bFIXME\b", "FIXME found"),
    (r"\bHACK\b", "HACK found"),
    (r"\bWIP\b", "WIP found"),
    (r"\bSLOP\b", "SLOP found"),
    (r"\bfoo\b", "Metasyntactic variable 'foo' found"),
    (r"\bbar\b", "Metasyntactic variable 'bar' found"),
    (r"\bbaz\b", "Metasyntactic variable 'baz' found"),
    (r"print\(", "Raw print() found (use logging or specific harness printers)"),
    # Casual language check (simple heuristics)
    (r"\boops\b", "Casual language 'oops'"),
    (r"\bwhoops\b", "Casual language 'whoops'"),
    (r"\bstuff\b", "Vague language 'stuff'"),
    (r"\bthing\b", "Vague language 'thing'"),
]

def check_file(filepath: str) -> List[str]:
    violations = []
    if not os.path.exists(filepath):
        return [f"File not found: {filepath}"]

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            content = "".join(lines)
    except Exception as e:
        return [f"Could not read file: {e}"]

    # 1. Check Banned Patterns
    for i, line in enumerate(lines):
        for pattern, reason in BANNED_PATTERNS:
            # Allow 'print' in test files or scripts explicitly designed for output (like this one)
            if "print(" in pattern and ("test_" in filepath or "tools/" in filepath):
                continue
                
            if re.search(pattern, line, re.IGNORECASE):
                # Skip if it looks like it's in the Vibe Check script itself (if we scanned it)
                if "BANNED_PATTERNS" in line: 
                    continue
                # Whitelist "Green Bar" (common term in CI/status)
                if "Green Bar" in line:
                    continue
                violations.append(f"Line {i+1}: {reason} (Content: {line.strip()})")

    # 2. Check Docstrings (Python only)
    if filepath.endswith(".py"):
        # Simple check: does the file start with a docstring?
        # We look for triple quotes at the start of the file content (ignoring shebang/imports for a rough check)
        # This is a heuristic. A proper AST check would be better but this is V1.
        if not (content.strip().startswith('"""') or content.strip().startswith("'''") or 
                (content.strip().startswith('#!') and ('"""' in content or "'''" in content))):
            violations.append("Module Missing Docstring (or not at top)")

    return violations

def main():
    print("MathLedger Vibe Check")
    print("=====================")
    print(f"Scanning {len(TARGETS)} files in First Organism path...")
    print("")

    total_violations = 0
    failed_files = 0

    for target in TARGETS:
        # Handle if target is a directory (not used yet but good for future)
        if os.path.isdir(target):
            # skip for now, we specified files
            continue
            
        violations = check_file(target)
        if violations:
            failed_files += 1
            total_violations += len(violations)
            print(f"FAIL: {target}")
            for v in violations:
                print(f"  - {v}")
        else:
            # Only print pass if verbose? No, let's just show failures to keep it clean,
            # or show a dot for progress.
            if os.path.exists(target):
                print(f"PASS: {target}")
            else:
                 print(f"SKIP: {target} (not found)")

    print("")
    if total_violations > 0:
        print(f"VIBE CHECK FAILED: {total_violations} violations in {failed_files} files.")
        sys.exit(1)
    else:
        print("VIBE CHECK PASSED. The organism is clean.")
        sys.exit(0)

if __name__ == "__main__":
    main()
