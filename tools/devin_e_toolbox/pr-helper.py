#!/usr/bin/env python3
"""
PR Helper - Automate PR creation with proper formatting and checks

Integrates with Cursor P (GitOps Conductor) for coordinated PR workflows.

Usage:
    python pr-helper.py create --title "feat: add feature" --tag POA
    python pr-helper.py check                    # Run pre-PR checks
    python pr-helper.py template --tag ASD       # Generate PR template
    python pr-helper.py cursor-sync              # Sync with Cursor P conductor
    python pr-helper.py cursor-status            # Check Cursor P status
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

DIFFERENTIATOR_TAGS = {
    'POA': 'Proof of Automation',
    'ASD': 'Algorithmic Superiority Demonstration',
    'RC': 'Reliability & Correctness',
    'ME': 'Metrics & Evidence',
    'IVL': 'Integration & Validation Layer',
    'NSF': 'Network Security & Forensics',
    'FM': 'Formal Methods'
}

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
        print(f"Error: Command failed")
        print(f"STDERR: {result.stderr}")
        sys.exit(result.returncode)
    return result

def check_branch_name():
    """Check if current branch follows naming convention"""
    result = run_command("git branch --show-current", check=False)
    branch = result.stdout.strip()
    
    pattern = r'^(feature|perf|ops|qa|devxp|docs)/[a-z0-9][a-z0-9\-]+$'
    if not re.match(pattern, branch):
        print(f"Warning: Branch name '{branch}' does not follow convention")
        print("Expected: (feature|perf|ops|qa|devxp|docs)/name-with-hyphens")
        return False
    return True

def check_ascii_compliance():
    """Check ASCII compliance in docs and scripts"""
    print("Checking ASCII compliance...")
    result = run_command("python tools/check_ascii.py", check=False)
    if result.returncode != 0:
        print("ASCII compliance check failed")
        return False
    print("ASCII compliance OK")
    return True

def check_tests():
    """Run unit tests"""
    print("Running unit tests...")
    result = run_command("NO_NETWORK=true pytest -q -k 'not integration and not derive_cli'", check=False)
    if result.returncode != 0:
        print("Unit tests failed")
        return False
    print("Unit tests OK")
    return True

def check_no_artifacts():
    """Check for uncommitted artifacts"""
    result = run_command("git status --porcelain", check=False)
    lines = result.stdout.strip().split('\n')
    
    blocked_patterns = ['artifacts/', 'dist/', '.pyc', '__pycache__']
    blocked_files = []
    
    for line in lines:
        if not line:
            continue
        file_path = line[3:]  # Skip status prefix
        if any(pattern in file_path for pattern in blocked_patterns):
            blocked_files.append(file_path)
    
    if blocked_files:
        print("Error: Attempting to commit blocked files:")
        for f in blocked_files:
            print(f"  - {f}")
        print("\nPlease remove these files from staging")
        return False
    
    return True

def generate_pr_template(tag):
    """Generate PR description template"""
    if tag not in DIFFERENTIATOR_TAGS:
        print(f"Error: Unknown tag '{tag}'")
        print(f"Available tags: {', '.join(DIFFERENTIATOR_TAGS.keys())}")
        sys.exit(1)
    
    template = f"""## Summary

[Brief description of changes and rationale]


**Differentiator**: [{tag}] - {DIFFERENTIATOR_TAGS[tag]}
**Acquisition Narrative**: [How this advances competitive positioning]
**Measurable Outcomes**: [Specific metrics or capabilities gained]
**Doctrine Alignment**: [Reference to core technical doctrine]


**Components Modified**:
- [List modified components]

**Risk Level**: [Low/Medium/High]

**Rollback Plan**: [How to revert if needed]


**Unit Tests**:
- [List unit test coverage]

**Integration Tests**:
- [List integration test coverage]

**Performance Validation**:
- [Performance impact assessment]


[List any files that overlap with other PRs]


- [ ] No artifacts/ or dist/ directories
- [ ] ASCII-only logs and documentation
- [ ] Network-free tests
- [ ] All tests passing
- [ ] Coverage floor maintained
- [ ] Branch naming convention followed
- [ ] Strategic differentiator tag included
- [ ] Doctrine alignment documented


https://app.devin.ai/sessions/90defe8017374d469a2fd6ffe57e352e


helpful.dolphin@pm.me (@helpfuldolphin)
"""
    
    return template

def cmd_check(args):
    """Run pre-PR checks"""
    print("Running pre-PR checks...")
    print()
    
    checks = [
        ("Branch naming", check_branch_name),
        ("ASCII compliance", check_ascii_compliance),
        ("No artifacts", check_no_artifacts),
        ("Unit tests", check_tests)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"=== {name} ===")
        result = check_func()
        results.append((name, result))
        print()
    
    print("=== Summary ===")
    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\nAll checks passed! Ready to create PR.")
        return 0
    else:
        print("\nSome checks failed. Please fix before creating PR.")
        return 1

def cmd_template(args):
    """Generate PR template"""
    template = generate_pr_template(args.tag)
    
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(template)
        print(f"PR template written to {output_path}")
    else:
        print(template)

def cmd_cursor_sync(args):
    """Sync with Cursor P GitOps Conductor"""
    import json
    print("=== Cursor P GitOps Sync ===")
    print()
    
    cursor_file = REPO_ROOT / '.cursor' / 'gitops.json'
    
    if not cursor_file.exists():
        print("No Cursor P coordination file found")
        print(f"Expected: {cursor_file}")
        print()
        print("Creating default coordination file...")
        cursor_file.parent.mkdir(parents=True, exist_ok=True)
        default_config = {
            "conductor": "cursor-p",
            "agents": ["devin-e"],
            "coordination_mode": "sequential",
            "pr_strategy": "single-pr-per-agent"
        }
        cursor_file.write_text(json.dumps(default_config, indent=2))
        print(f"Created: {cursor_file}")
    else:
        print(f"Found Cursor P config: {cursor_file}")
        config = json.loads(cursor_file.read_text())
        print(f"Conductor: {config.get('conductor', 'unknown')}")
        print(f"Agents: {', '.join(config.get('agents', []))}")
        print(f"Mode: {config.get('coordination_mode', 'unknown')}")
    
    print()
    print("Sync complete")

def cmd_cursor_status(args):
    """Check Cursor P coordination status"""
    import json
    print("=== Cursor P Status ===")
    print()
    
    result = run_command("git branch --show-current", check=False)
    branch = result.stdout.strip()
    print(f"Current branch: {branch}")
    
    result = run_command("gh pr list --state open", check=False)
    if result.returncode == 0 and result.stdout.strip():
        print()
        print("Open PRs:")
        print(result.stdout)
    else:
        print("No open PRs")
    
    cursor_file = REPO_ROOT / '.cursor' / 'gitops.json'
    if cursor_file.exists():
        print()
        print(f"Cursor P config: {cursor_file}")
    else:
        print()
        print("No Cursor P coordination file")

def cmd_create(args):
    """Create PR with proper formatting"""
    if args.tag not in DIFFERENTIATOR_TAGS:
        print(f"Error: Unknown tag '{args.tag}'")
        print(f"Available tags: {', '.join(DIFFERENTIATOR_TAGS.keys())}")
        sys.exit(1)
    
    print("Running pre-PR checks...")
    if cmd_check(args) != 0:
        print("\nPre-PR checks failed. Fix issues before creating PR.")
        sys.exit(1)
    
    result = run_command("git branch --show-current")
    branch = result.stdout.strip()
    
    title = f"[{args.tag}] {args.title}"
    
    body = generate_pr_template(args.tag)
    
    body_file = REPO_ROOT / '.pr_body.md'
    body_file.write_text(body)
    
    print(f"\nCreating PR:")
    print(f"  Branch: {branch}")
    print(f"  Title: {title}")
    print(f"  Tag: [{args.tag}] {DIFFERENTIATOR_TAGS[args.tag]}")
    print()
    
    cmd = f'gh pr create --title "{title}" --body-file .pr_body.md --base main'
    if args.draft:
        cmd += ' --draft'
    
    result = run_command(cmd, check=False)
    
    body_file.unlink()
    
    if result.returncode == 0:
        print("\nPR created successfully!")
        print(result.stdout)
        
        cursor_file = REPO_ROOT / '.cursor' / 'gitops.json'
        if cursor_file.exists():
            print()
            print("Syncing with Cursor P...")
            cmd_cursor_sync(args)
    else:
        print("\nError creating PR:")
        print(result.stderr)
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='PR Helper - Automate PR creation')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    check_parser = subparsers.add_parser('check', help='Run pre-PR checks')
    
    template_parser = subparsers.add_parser('template', help='Generate PR template')
    template_parser.add_argument('--tag', required=True, choices=DIFFERENTIATOR_TAGS.keys(), help='Strategic differentiator tag')
    template_parser.add_argument('--output', help='Output file path')
    
    create_parser = subparsers.add_parser('create', help='Create PR')
    create_parser.add_argument('--title', required=True, help='PR title (without tag prefix)')
    create_parser.add_argument('--tag', required=True, choices=DIFFERENTIATOR_TAGS.keys(), help='Strategic differentiator tag')
    create_parser.add_argument('--draft', action='store_true', help='Create as draft PR')
    
    cursor_sync_parser = subparsers.add_parser('cursor-sync', help='Sync with Cursor P GitOps Conductor')
    cursor_status_parser = subparsers.add_parser('cursor-status', help='Check Cursor P coordination status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    commands = {
        'check': cmd_check,
        'template': cmd_template,
        'create': cmd_create,
        'cursor-sync': cmd_cursor_sync,
        'cursor-status': cmd_cursor_status
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == '__main__':
    main()
