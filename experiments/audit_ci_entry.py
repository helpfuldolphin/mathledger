"""
CI-friendly entry point for uplift_u2 Evidence Pack audits.

This wrapper provides a minimal, scriptable interface suitable for
GitHub Actions or other CI systems. It invokes audit_uplift_u2_all.py
and provides a concise stdout summary.

Exit codes (forwarded from audit_uplift_u2_all.py):
- 0: PASS - All experiments passed integrity checks
- 1: FAIL - One or more experiments failed
- 2: MIXED - Some experiments have missing artifacts

PHASE II — NOT USED IN PHASE I

---

## CI Integration Example (GitHub Actions)

Add this to your `.github/workflows/audit.yml`:

```yaml
name: Audit Evidence Packs

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  audit:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e .
    
    - name: Audit Evidence Packs
      run: |
        python experiments/audit_ci_entry.py results/uplift_u2
```

---

## Usage Examples

Basic usage (defaults to results/uplift_u2):
```bash
python experiments/audit_ci_entry.py
```

Specify custom directory:
```bash
python experiments/audit_ci_entry.py path/to/experiments
```

Save reports:
```bash
python experiments/audit_ci_entry.py results/uplift_u2 \\
  --output-json ci_audit.json \\
  --output-md ci_audit.md
```

---
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """CI entry point for Evidence Pack audits."""
    parser = argparse.ArgumentParser(
        description="CI-friendly wrapper for uplift_u2 Evidence Pack audits.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    parser.add_argument(
        "root_dir",
        type=str,
        default="results/uplift_u2",
        nargs='?',
        help="Root directory containing experiments (default: results/uplift_u2)"
    )
    
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write JSON report to file"
    )
    
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write Markdown report to file"
    )
    
    args = parser.parse_args()
    
    # Construct command for audit_uplift_u2_all.py
    script_path = Path(__file__).parent / "audit_uplift_u2_all.py"
    
    cmd = [sys.executable, str(script_path), args.root_dir]
    
    if args.output_json:
        cmd.extend(["--output-json", args.output_json])
    
    if args.output_md:
        cmd.extend(["--output-md", args.output_md])
    
    # Run the audit
    print("=" * 60)
    print("CI EVIDENCE PACK AUDIT")
    print("=" * 60)
    print(f"Target: {args.root_dir}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print captured output
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    
    # Parse summary from stdout (extract key metrics)
    # The audit_uplift_u2_all.py prints a summary section
    lines = result.stdout.split('\n')
    
    # Extract summary statistics
    total_exp = None
    passed = None
    failed = None
    missing = None
    
    for line in lines:
        if "Total Experiments:" in line:
            try:
                total_exp = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Passed:" in line:
            try:
                passed = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Failed:" in line:
            try:
                failed = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Missing:" in line:
            try:
                missing = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
    
    # Print CI-friendly summary line
    print()
    print("=" * 60)
    if total_exp is not None:
        print(f"Summary: {total_exp} experiment(s) audited")
        if passed is not None:
            print(f"  ✅ Passed: {passed}")
        if failed is not None and failed > 0:
            print(f"  ❌ Failed: {failed}")
        if missing is not None and missing > 0:
            print(f"  ⚠️  Missing: {missing}")
    else:
        print("Summary: Unable to parse audit results")
    
    print("=" * 60)
    
    # Exit with the same code as the underlying audit
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
