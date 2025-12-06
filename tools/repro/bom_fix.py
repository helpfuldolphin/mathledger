#!/usr/bin/env python3
"""
BOM Scanner and Fixer - Detect and remove UTF-8 BOM from Python files.

UTF-8 BOM (Byte Order Mark) is the byte sequence 0xEF 0xBB 0xBF at the start
of files. It causes Python syntax errors and prevents drift_sentinel.py from
scanning files, creating blind spots in determinism enforcement.

This tool:
1. Scans Python files for UTF-8 BOM
2. Re-writes files as UTF-8 without BOM
3. Logs each fixed file
4. Produces summary report

Usage:
    python tools/repro/bom_fix.py --scan backend/
    python tools/repro/bom_fix.py --fix backend/
    python tools/repro/bom_fix.py --verify backend/

Exit Codes:
    0: Success (no BOM files found or all fixed)
    1: BOM files found (scan mode) or fix failed
    2: Verification failed (BOM files remain after fix)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

UTF8_BOM = b'\xef\xbb\xbf'


def get_repo_root() -> Path:
    """Get repository root directory."""
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (no .git directory)")


def has_bom(file_path: Path) -> bool:
    """Check if file starts with UTF-8 BOM."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(3)
            return header == UTF8_BOM
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return False


def remove_bom(file_path: Path) -> bool:
    """Remove UTF-8 BOM from file."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        if content.startswith(UTF8_BOM):
            content_without_bom = content[3:]
            
            with open(file_path, 'wb') as f:
                f.write(content_without_bom)
            
            return True
        else:
            return False
    
    except Exception as e:
        print(f"Error: Could not fix {file_path}: {e}", file=sys.stderr)
        return False


def scan_directory(directory: Path) -> List[Path]:
    """Scan directory for Python files with BOM."""
    bom_files = []
    
    for py_file in directory.rglob('*.py'):
        if '__pycache__' in py_file.parts or '.venv' in py_file.parts:
            continue
        
        if has_bom(py_file):
            bom_files.append(py_file)
    
    return bom_files


def fix_directory(directory: Path) -> Tuple[List[Path], List[Path]]:
    """Fix all Python files with BOM in directory."""
    bom_files = scan_directory(directory)
    
    fixed = []
    failed = []
    
    for file_path in bom_files:
        if remove_bom(file_path):
            fixed.append(file_path)
            print(f"Fixed: {file_path}")
        else:
            failed.append(file_path)
            print(f"Failed: {file_path}", file=sys.stderr)
    
    return fixed, failed


def verify_directory(directory: Path) -> List[Path]:
    """Verify no BOM files remain in directory."""
    return scan_directory(directory)


def main():
    parser = argparse.ArgumentParser(
        description="BOM Scanner and Fixer - Remove UTF-8 BOM from Python files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/repro/bom_fix.py --scan backend/
  
  python tools/repro/bom_fix.py --fix backend/
  
  python tools/repro/bom_fix.py --verify backend/
        """
    )
    
    parser.add_argument(
        "--scan",
        type=str,
        metavar="DIR",
        help="Scan directory for BOM files"
    )
    
    parser.add_argument(
        "--fix",
        type=str,
        metavar="DIR",
        help="Fix all BOM files in directory"
    )
    
    parser.add_argument(
        "--verify",
        type=str,
        metavar="DIR",
        help="Verify no BOM files remain in directory"
    )
    
    args = parser.parse_args()
    
    if not (args.scan or args.fix or args.verify):
        parser.print_help()
        return 1
    
    try:
        repo_root = get_repo_root()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    
    if args.scan:
        print("=" * 80)
        print("SCANNING FOR BOM FILES")
        print("=" * 80)
        
        scan_dir = Path(args.scan)
        if not scan_dir.is_absolute():
            scan_dir = repo_root / scan_dir
        
        if not scan_dir.exists():
            print(f"ERROR: Directory not found: {scan_dir}", file=sys.stderr)
            return 1
        
        bom_files = scan_directory(scan_dir)
        
        if bom_files:
            print(f"\n[FAIL] Found {len(bom_files)} files with UTF-8 BOM:")
            for file_path in bom_files:
                rel_path = file_path.relative_to(repo_root)
                print(f"  - {rel_path}")
            print("\nRun with --fix to remove BOM from these files.")
            return 1
        else:
            print("\n[PASS] No BOM files found")
            return 0
    
    elif args.fix:
        print("=" * 80)
        print("FIXING BOM FILES")
        print("=" * 80)
        
        fix_dir = Path(args.fix)
        if not fix_dir.is_absolute():
            fix_dir = repo_root / fix_dir
        
        if not fix_dir.exists():
            print(f"ERROR: Directory not found: {fix_dir}", file=sys.stderr)
            return 1
        
        fixed, failed = fix_directory(fix_dir)
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if fixed:
            print(f"\n[PASS] BOM Purge: {len(fixed)} files fixed")
            for file_path in fixed:
                rel_path = file_path.relative_to(repo_root)
                print(f"  - {rel_path}")
        
        if failed:
            print(f"\n[FAIL] Failed to fix {len(failed)} files:")
            for file_path in failed:
                rel_path = file_path.relative_to(repo_root)
                print(f"  - {rel_path}")
            return 1
        
        if not fixed and not failed:
            print("\n[PASS] No BOM files found")
        
        return 0
    
    elif args.verify:
        print("=" * 80)
        print("VERIFYING NO BOM FILES REMAIN")
        print("=" * 80)
        
        verify_dir = Path(args.verify)
        if not verify_dir.is_absolute():
            verify_dir = repo_root / verify_dir
        
        if not verify_dir.exists():
            print(f"ERROR: Directory not found: {verify_dir}", file=sys.stderr)
            return 1
        
        bom_files = verify_directory(verify_dir)
        
        if bom_files:
            print(f"\n[FAIL] Verification failed: {len(bom_files)} BOM files remain:")
            for file_path in bom_files:
                rel_path = file_path.relative_to(repo_root)
                print(f"  - {rel_path}")
            return 2
        else:
            print("\n[PASS] Verification passed: 0 BOM files remain")
            return 0


if __name__ == "__main__":
    sys.exit(main())
