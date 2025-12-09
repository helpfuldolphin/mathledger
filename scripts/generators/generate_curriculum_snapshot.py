#!/usr/bin/env python3
"""
Curriculum Snapshot Generator

Generates a deterministic, versioned snapshot of the MathLedger curriculum.

Data Sources:
  - curriculum/problems/**/*.md (problem definitions with YAML frontmatter)
  - curriculum/topics.yml (topic taxonomy)

Output:
  - JSON snapshot compliant with schemas/curriculum_snapshot.schema.json
  - Printed to stdout in RFC 8785 canonical form

Exit Codes:
  0 - Success
  1 - Data source missing or corrupted
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML library not installed", file=sys.stderr)
    print("Install with: pip3 install pyyaml", file=sys.stderr)
    sys.exit(1)


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def parse_markdown_with_frontmatter(filepath: Path) -> Dict[str, Any]:
    """
    Parse a Markdown file with YAML frontmatter.
    
    Returns a dict with 'metadata' and 'content' keys.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for frontmatter delimiters
    if not content.startswith('---\n'):
        raise ValueError(f"File {filepath} does not have YAML frontmatter")
    
    # Find the closing delimiter
    parts = content.split('---\n', 2)
    if len(parts) < 3:
        raise ValueError(f"File {filepath} has malformed frontmatter")
    
    frontmatter_str = parts[1]
    markdown_content = parts[2]
    
    # Parse YAML frontmatter
    metadata = yaml.safe_load(frontmatter_str)
    
    return {
        'metadata': metadata,
        'content': markdown_content
    }


def generate_curriculum_snapshot(repo_root: Path) -> Dict[str, Any]:
    """Generate the curriculum snapshot."""
    
    # Load topic taxonomy
    topics_file = repo_root / 'curriculum' / 'topics.yml'
    if not topics_file.exists():
        print(f"ERROR: Topic taxonomy file not found: {topics_file}", file=sys.stderr)
        sys.exit(1)
    
    with open(topics_file, 'r', encoding='utf-8') as f:
        topic_taxonomy = yaml.safe_load(f)
    
    # Find all problem definition files
    problems_dir = repo_root / 'curriculum' / 'problems'
    if not problems_dir.exists():
        print(f"ERROR: Problems directory not found: {problems_dir}", file=sys.stderr)
        sys.exit(1)
    
    problem_files = list(problems_dir.glob('**/*.md'))
    
    # Extract problem metadata
    problems = []
    for problem_file in sorted(problem_files):
        try:
            parsed = parse_markdown_with_frontmatter(problem_file)
            metadata = parsed['metadata']
            content = parsed['content']
            
            # Compute content hash
            content_hash = compute_content_hash(content)
            
            # Build problem entry
            problem_entry = {
                'id': metadata.get('id', ''),
                'title': metadata.get('title', ''),
                'topic': metadata.get('topic', ''),
                'difficulty_score': float(metadata.get('difficulty_score', 0.0)),
                'content_hash': content_hash
            }
            
            problems.append(problem_entry)
        
        except Exception as e:
            print(f"ERROR: Failed to parse {problem_file}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Build snapshot
    snapshot = {
        'version': '1.0.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'topic_taxonomy': topic_taxonomy,
        'problems': problems
    }
    
    return snapshot


def canonicalize_json(obj: Any) -> str:
    """
    Serialize an object to RFC 8785 canonical JSON.
    
    This is a simplified implementation. For production, use a library
    like `canonicaljson` for full RFC 8785 compliance.
    """
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(',', ':'))


def main():
    repo_root = Path.cwd()
    
    # Generate snapshot
    snapshot = generate_curriculum_snapshot(repo_root)
    
    # Canonicalize and print
    canonical_json = canonicalize_json(snapshot)
    print(canonical_json)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
