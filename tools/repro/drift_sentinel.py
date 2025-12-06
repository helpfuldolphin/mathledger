#!/usr/bin/env python3
"""
Drift Sentinel - Detect nondeterministic operations in code.

This sentinel scans Python files for calls to nondeterministic functions
(time.time, datetime.utcnow, uuid.uuid4, np.random) outside approved wrappers.

Usage:
    python tools/repro/drift_sentinel.py [files...]
    python tools/repro/drift_sentinel.py --all
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


NONDETERMINISTIC_PATTERNS = {
    'time.time': 'Use deterministic_unix_timestamp() from backend.repro.determinism',
    'datetime.utcnow': 'Use deterministic_timestamp() from backend.repro.determinism',
    'datetime.datetime.utcnow': 'Use deterministic_timestamp() from backend.repro.determinism',
    'datetime.now': 'Use deterministic_timestamp() from backend.repro.determinism',
    'datetime.datetime.now': 'Use deterministic_timestamp() from backend.repro.determinism',
    'uuid.uuid4': 'Use deterministic_uuid() from backend.repro.determinism',
    'uuid.UUID': 'Use deterministic_uuid() from backend.repro.determinism',
    'np.random.random': 'Use SeededRNG() from backend.repro.determinism',
    'np.random.rand': 'Use SeededRNG() from backend.repro.determinism',
    'np.random.randn': 'Use SeededRNG() from backend.repro.determinism',
    'np.random.randint': 'Use SeededRNG() from backend.repro.determinism',
    'random.random': 'Use SeededRNG() from backend.repro.determinism',
    'random.randint': 'Use SeededRNG() from backend.repro.determinism',
}


def load_whitelist(whitelist_path: Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Load whitelist of approved files and functions from JSON.
    
    Returns:
        (file_whitelist, function_whitelist)
        function_whitelist maps file -> set of qualified function names
    """
    if not whitelist_path.exists():
        return set(), {}
    
    with open(whitelist_path, 'r') as f:
        data = json.load(f)
        file_whitelist = set(data.get('whitelist', []))
        function_whitelist = {}
        
        for entry in data.get('function_whitelist', []):
            file = entry.get('file')
            qualname = entry.get('qualname')
            if file and qualname:
                if file not in function_whitelist:
                    function_whitelist[file] = set()
                function_whitelist[file].add(qualname)
        
        return file_whitelist, function_whitelist


class NondeterminismDetector(ast.NodeVisitor):
    """AST visitor to detect nondeterministic function calls with function-scope whitelist."""
    
    def __init__(self, filepath: str, function_whitelist: Set[str]):
        self.filepath = filepath
        self.function_whitelist = function_whitelist
        self.violations = []
        self.current_function = None
        self.function_stack = []
        self.deterministic_ok_functions = set()
    
    def visit_FunctionDef(self, node):
        """Track function context and check for @deterministic_ok decorator."""
        qualname = self._get_qualname(node.name)
        self.function_stack.append(qualname)
        self.current_function = qualname
        
        if self._has_deterministic_ok_decorator(node):
            self.deterministic_ok_functions.add(qualname)
        
        self.generic_visit(node)
        
        self.function_stack.pop()
        self.current_function = self.function_stack[-1] if self.function_stack else None
    
    def visit_AsyncFunctionDef(self, node):
        """Track async function context."""
        self.visit_FunctionDef(node)
    
    def visit_Call(self, node):
        """Check function calls for nondeterministic patterns."""
        call_str = self._get_call_string(node)
        
        if call_str in NONDETERMINISTIC_PATTERNS:
            if not self._is_whitelisted():
                self.violations.append({
                    'file': self.filepath,
                    'qualname': self.current_function or '<module>',
                    'line': node.lineno,
                    'column': node.col_offset,
                    'pattern': call_str,
                    'recommendation': NONDETERMINISTIC_PATTERNS[call_str]
                })
        
        self.generic_visit(node)
    
    def _is_whitelisted(self) -> bool:
        """Check if current function is whitelisted."""
        if self.current_function:
            if self.current_function in self.function_whitelist:
                return True
            if self.current_function in self.deterministic_ok_functions:
                return True
        return False
    
    def _get_qualname(self, name: str) -> str:
        """Get qualified name for function."""
        if self.function_stack:
            return f"{self.function_stack[-1]}.{name}"
        return name
    
    def _has_deterministic_ok_decorator(self, node) -> bool:
        """Check if function has @deterministic_ok decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'deterministic_ok':
                return True
            if isinstance(decorator, ast.Attribute) and decorator.attr == 'deterministic_ok':
                return True
        return False
    
    def _get_call_string(self, node) -> str:
        """Extract the full call string from an AST Call node."""
        if isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        elif isinstance(node.func, ast.Name):
            return node.func.id
        return ''


def scan_file(filepath: Path, function_whitelist: Set[str] = None) -> List[Dict]:
    """
    Scan a single Python file for nondeterministic operations.
    
    Args:
        filepath: Path to Python file
        function_whitelist: Set of qualified function names to whitelist
    
    Returns:
        List of violations found
    """
    if function_whitelist is None:
        function_whitelist = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=str(filepath))
        detector = NondeterminismDetector(str(filepath), function_whitelist)
        detector.visit(tree)
        return detector.violations
    
    except SyntaxError as e:
        print(f"Warning: Syntax error in {filepath}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Warning: Error scanning {filepath}: {e}", file=sys.stderr)
        return []


def scan_directory(directory: Path, file_whitelist: Set[str], function_whitelist_map: Dict[str, Set[str]]) -> List[Dict]:
    """
    Recursively scan directory for Python files.
    
    Args:
        directory: Root directory to scan
        file_whitelist: Set of file paths to skip entirely
        function_whitelist_map: Map of file -> set of whitelisted function qualnames
    
    Returns:
        List of all violations found
    """
    all_violations = []
    
    for py_file in directory.rglob('*.py'):
        rel_path = str(py_file.relative_to(directory.parent))
        if rel_path in file_whitelist:
            continue
        
        if '__pycache__' in py_file.parts or '.venv' in py_file.parts:
            continue
        
        func_whitelist = function_whitelist_map.get(rel_path, set())
        violations = scan_file(py_file, func_whitelist)
        all_violations.extend(violations)
    
    return all_violations


def generate_drift_report(violations: List[Dict], output_path: Path):
    """Generate drift report JSON."""
    report = {
        "version": "1.0.0",
        "status": "DRIFT_DETECTED" if violations else "CLEAN",
        "timestamp": "2025-10-19T00:00:00Z",
        "violation_count": len(violations),
        "violations": violations,
        "patterns_checked": list(NONDETERMINISTIC_PATTERNS.keys()),
        "recommendation": "Replace nondeterministic calls with deterministic helpers",
        "playbook": "docs/repro/DRIFT_RESPONSE_PLAYBOOK.md"
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, sort_keys=True)
    
    print(f"Drift report written to: {output_path}")


def generate_drift_patch(violations: List[Dict], output_path: Path):
    """Generate suggested patch for fixing violations."""
    if not violations:
        return
    
    patch_lines = [
        "# Drift Sentinel - Suggested Fixes",
        "# Apply these changes to eliminate nondeterministic operations",
        "",
    ]
    
    for violation in violations:
        patch_lines.append(f"# File: {violation['file']}:{violation['line']}")
        patch_lines.append(f"# Pattern: {violation['pattern']}")
        patch_lines.append(f"# Fix: {violation['recommendation']}")
        patch_lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(patch_lines))
    
    print(f"Drift patch written to: {output_path}")


def get_staged_files() -> List[Path]:
    """Get list of staged Python files from git."""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
            capture_output=True,
            text=True,
            check=True
        )
        files = [Path(f) for f in result.stdout.strip().split('\n') if f.endswith('.py')]
        return files
    except subprocess.CalledProcessError:
        return []


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Drift Sentinel - Detect nondeterministic operations'
    )
    parser.add_argument('files', nargs='*', help='Files to scan')
    parser.add_argument('--all', action='store_true', help='Scan entire backend directory')
    parser.add_argument('--staged', action='store_true', help='Scan only git-staged files (for pre-commit)')
    parser.add_argument('--whitelist', type=str, default='artifacts/repro/drift_whitelist.json',
                        help='Path to whitelist JSON')
    parser.add_argument('--report', type=str, default='artifacts/repro/drift_report.json',
                        help='Path to output drift report')
    parser.add_argument('--patch', type=str, default='artifacts/repro/drift_patch.diff',
                        help='Path to output drift patch')
    args = parser.parse_args(argv)
    
    whitelist_path = Path(args.whitelist)
    file_whitelist, function_whitelist_map = load_whitelist(whitelist_path)
    print(f"Loaded whitelist: {len(file_whitelist)} files, {sum(len(v) for v in function_whitelist_map.values())} functions")
    
    all_violations = []
    
    if args.staged:
        staged_files = get_staged_files()
        if not staged_files:
            print("No staged Python files to scan.")
            return 0
        print(f"Scanning {len(staged_files)} staged files...")
        for path in staged_files:
            if path.exists() and path.suffix == '.py':
                rel_path = str(path)
                func_whitelist = function_whitelist_map.get(rel_path, set())
                violations = scan_file(path, func_whitelist)
                all_violations.extend(violations)
    elif args.all:
        print("Scanning entire backend directory...")
        backend_dir = Path('backend')
        if backend_dir.exists():
            all_violations = scan_directory(backend_dir, file_whitelist, function_whitelist_map)
    elif args.files:
        print(f"Scanning {len(args.files)} files...")
        for filepath in args.files:
            path = Path(filepath)
            if path.exists() and path.suffix == '.py':
                rel_path = str(path)
                func_whitelist = function_whitelist_map.get(rel_path, set())
                violations = scan_file(path, func_whitelist)
                all_violations.extend(violations)
    else:
        print("No files specified. Use --all to scan entire backend, --staged for git-staged files, or provide file paths.")
        return 1
    
    # Report results
    print()
    print("=" * 70)
    if all_violations:
        print(f"[FAIL] Drift Sentinel: {len(all_violations)} violations detected")
        print("=" * 70)
        print()
        
        for violation in all_violations:
            print(f"  {violation['file']}:{violation['qualname']}:{violation['line']}")
            print(f"    Pattern: {violation['pattern']}")
            print(f"    Fix: {violation['recommendation']}")
            print()
        
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_drift_report(all_violations, report_path)
        
        patch_path = Path(args.patch)
        generate_drift_patch(all_violations, patch_path)
        
        print("ABSTAIN: Nondeterministic operations detected. Review drift report.")
        return 1
    else:
        print("[PASS] Drift Sentinel: No violations detected")
        print("All code uses deterministic helpers")
        print("=" * 70)
        
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_drift_report(all_violations, report_path)
        
        return 0


if __name__ == '__main__':
    sys.exit(main())
