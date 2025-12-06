#!/usr/bin/env python3
"""
Determinism Gate CLI
====================

Guardian of the "Given S + I ⇒ A is Unique" guarantee.

This script enforces the Determinism Contract by:
1. Verifying the First Organism harness produces identical outputs for the same seed.
2. Running the RFL Runner integration test to ensure artifact bitwise reproducibility.
3. Scanning the codebase for forbidden entropy primitives in critical paths.

Usage:
    python scripts/determinism_gate.py
"""

import sys
import os
import subprocess
import re
from pathlib import Path
from typing import List, Tuple

# Set dummy env vars for tests
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mathledger")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

# Forbidden primitives pattern
FORBIDDEN_PATTERNS = [
    (r"datetime\.now\(", "Wall-clock time forbidden"),
    (r"datetime\.utcnow\(", "Wall-clock time forbidden"),
    (r"time\.time\(", "Wall-clock time forbidden (use deterministic_unix_timestamp)"),
    (r"uuid\.uuid4\(", "Random UUID forbidden (use deterministic_uuid)"),
    # Catch random usage but allow seeding and beta/poisson in bootstrap stats (legacy, to be fixed)
    (r"(?<!np\.)random\.(?!seed)", "Unseeded global random forbidden (use SeededRNG)"),
    (r"np\.random\.(?!seed|RandomState|beta|poisson)", "Unseeded numpy random forbidden (use SeededRNG)"),
    (r"os\.urandom", "OS entropy forbidden"),
]

# Directories to scan strictly
CRITICAL_PATHS = [
    "backend/repro",
    "backend/axiom_engine",
    "backend/ledger/ingest.py",  # Only ingest, not legacy consensus yet
    "rfl",
    "attestation",
]

# Exceptions (file_substring, line_content_pattern)
EXCEPTIONS = [
    ("backend/axiom_engine/derive_worker.py", r"now_op = time\.time\(\)"),
    ("backend/axiom_engine/derive_worker.py", r"last_seal_ts = time\.time\(\)"), 
    ("rfl/experiment.py", r"duration_seconds"), 
    ("backend/repro/determinism.py", r"datetime\.fromtimestamp"),
    ("backend/repro/determinism.py", r"random\.Random"),
    # Allow comments mentioning the forbidden terms
    ("", r"^\s*#"), 
    ("", r"^\s*\"\"\""),
    # Allow docstrings (heuristic)
    ("", r"^\s*-\s+NO\s+"), # Harness docstring
    ("", r"calls\s+in\s+the\s+attestation\s+path"), # Experiment docstring
]

def run_command(cmd: List[str], desc: str) -> bool:
    print(f"Running {desc}...")
    try:
        # Pass current env
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        if result.returncode == 0:
            print("  [PASS]")
            return True
        else:
            print("  [FAIL]")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def scan_codebase() -> bool:
    print("Scanning critical paths for forbidden primitives...")
    violations = []
    
    root_dir = Path(os.getcwd())
    
    for path_str in CRITICAL_PATHS:
        path = root_dir / path_str
        if not path.exists():
            # It might be a file
            if path_str.endswith(".py"):
                files_to_check = [path]
            else:
                continue
        else:
            if path.is_file():
                files_to_check = [path]
            else:
                files_to_check = path.rglob("*.py")

        for file_path in files_to_check:
            rel_path = file_path.relative_to(root_dir).as_posix()
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                continue
            
            for i, line in enumerate(lines):
                line_content = line.strip()
                
                # Skip comments
                if line_content.startswith("#") or line_content.startswith('"""'):
                    continue
                    
                for pattern, reason in FORBIDDEN_PATTERNS:
                    if re.search(pattern, line_content):
                        # Check exceptions
                        is_excepted = False
                        for exc_file, exc_pattern in EXCEPTIONS:
                            # Empty exc_file means global exception (like comments, though handled above)
                            if (exc_file == "" or exc_file in rel_path) and re.search(exc_pattern, line_content):
                                is_excepted = True
                                break
                        
                        if not is_excepted:
                            violations.append(f"{rel_path}:{i+1}: {reason} -> {line_content}")

    if violations:
        print("  [FAIL] Violations found:")
        for v in violations:
            print(f"    - {v}")
        return False
    else:
        print("  [PASS] No forbidden primitives found.")
        return True

def main():
    print("GEMINI C -- Determinism Gate")
    print("==================================")
    
    all_passed = True
    
    # 1. Verify First Organism Harness
    harness_test = ["uv", "run", "pytest", "tests/integration/test_first_organism_determinism.py"]
    if not run_command(harness_test, "First Organism Harness Verification"):
        all_passed = False
        
    # 2. Verify RFL Runner Bitwise Reproducibility
    rfl_test = ["uv", "run", "pytest", "tests/integration/test_rfl_runner_determinism.py"]
    if not run_command(rfl_test, "RFL Runner Bitwise Reproducibility"):
        all_passed = False
        
    # 3. Static Analysis
    if not scan_codebase():
        all_passed = False
        
    print("==================================")
    if all_passed:
        print("DETERMINISM AUDIT PASSED")
        sys.exit(0)
    else:
        print("DETERMINISM AUDIT FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
