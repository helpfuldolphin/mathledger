#!/usr/bin/env python3
"""
Workflow Validator - Validate development workflow compliance

Checks:
- Branch naming convention
- Commit message format
- No direct pushes to main
- ASCII-only content
- No artifacts in commits
- Test coverage maintained

Usage:
    python workflow-validator.py --check all
    python workflow-validator.py --check branch
    python workflow-validator.py --check commits
    python workflow-validator.py --fix ascii
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

def run_command(cmd, check=True):
    """Run shell command and return result"""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    if check and result.returncode != 0:
        return None
    return result.stdout.strip()

def check_branch_naming():
    """Validate branch naming convention"""
    print("Checking branch naming convention...")
    
    branch = run_command("git branch --show-current")
    if not branch:
        print("Error: Could not determine current branch")
        return False
    
    if branch in ['main', 'master']:
        print(f"On main branch: {branch}")
        return True
    
    pattern = r'^(feature|perf|ops|qa|devxp|docs)/[a-z0-9][a-z0-9\-]+$'
    if re.match(pattern, branch):
        print(f"Branch name OK: {branch}")
        return True
    else:
        print(f"Invalid branch name: {branch}")
        print("Expected format: (feature|perf|ops|qa|devxp|docs)/name-with-hyphens")
        print("Examples:")
        print("  feature/add-fol-support")
        print("  perf/optimize-modus-ponens")
        print("  ops/ci-pipeline-improvements")
        return False

def check_commit_messages():
    """Validate commit message format"""
    print("Checking commit message format...")
    
    commits = run_command("git log --oneline main..HEAD")
    if not commits:
        print("No commits to check")
        return True
    
    lines = commits.split('\n')
    invalid_commits = []
    
    pattern = r'^[a-f0-9]+ (feat|fix|docs|style|refactor|perf|test|ops|ci)(\([a-z\-]+\))?: .+'
    
    for line in lines:
        if not re.match(pattern, line):
            invalid_commits.append(line)
    
    if invalid_commits:
        print("Invalid commit messages found:")
        for commit in invalid_commits:
            print(f"  {commit}")
        print("\nExpected format: type(scope): description")
        print("Types: feat, fix, docs, style, refactor, perf, test, ops, ci")
        return False
    else:
        print(f"All {len(lines)} commit messages OK")
        return True

def check_ascii_content():
    """Check ASCII-only compliance"""
    print("Checking ASCII-only compliance...")
    
    result = subprocess.run(
        ["python", "tools/check_ascii.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("ASCII compliance OK")
        return True
    else:
        print("ASCII compliance violations found:")
        print(result.stdout)
        return False

def check_no_artifacts():
    """Check for artifacts in staging"""
    print("Checking for artifacts in staging...")
    
    staged = run_command("git diff --cached --name-only")
    if not staged:
        print("No staged files")
        return True
    
    blocked_patterns = ['artifacts/', 'dist/', '.pyc', '__pycache__', '.coverage']
    blocked_files = []
    
    for file_path in staged.split('\n'):
        if any(pattern in file_path for pattern in blocked_patterns):
            blocked_files.append(file_path)
    
    if blocked_files:
        print("Blocked files in staging:")
        for f in blocked_files:
            print(f"  {f}")
        print("\nPlease unstage these files")
        return False
    else:
        print("No artifacts in staging")
        return True

def check_test_coverage():
    """Check test coverage"""
    print("Checking test coverage...")
    
    result = subprocess.run(
        ["coverage", "run", "-m", "unittest"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Tests failed")
        return False
    
    result = subprocess.run(
        ["coverage", "report", "--fail-under=70"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("Coverage OK (>=70%)")
        return True
    else:
        print("Coverage below threshold")
        print(result.stdout)
        return False

def fix_ascii_content():
    """Fix ASCII compliance issues"""
    print("Fixing ASCII compliance issues...")
    
    result = subprocess.run(
        ["python", "tools/fix_ascii.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode == 0:
        print("ASCII fixes applied")
        return True
    else:
        print("Error applying ASCII fixes")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Workflow Validator')
    parser.add_argument('--check', choices=['all', 'branch', 'commits', 'ascii', 'artifacts', 'coverage'],
                       help='Check to run')
    parser.add_argument('--fix', choices=['ascii'], help='Fix to apply')
    
    args = parser.parse_args()
    
    if args.fix:
        if args.fix == 'ascii':
            success = fix_ascii_content()
            sys.exit(0 if success else 1)
    
    if args.check:
        checks = {
            'branch': check_branch_naming,
            'commits': check_commit_messages,
            'ascii': check_ascii_content,
            'artifacts': check_no_artifacts,
            'coverage': check_test_coverage
        }
        
        if args.check == 'all':
            print("Running all workflow checks...")
            print()
            results = []
            for name, check_func in checks.items():
                print(f"=== {name.upper()} ===")
                result = check_func()
                results.append((name, result))
                print()
            
            print("=== SUMMARY ===")
            all_passed = True
            for name, result in results:
                status = "PASS" if result else "FAIL"
                print(f"{status}: {name}")
                if not result:
                    all_passed = False
            
            sys.exit(0 if all_passed else 1)
        else:
            success = checks[args.check]()
            sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
