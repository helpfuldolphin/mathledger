#!/usr/bin/env python3
"""
Strategic PR Validator for enforcing differentiator tags and Strategic Impact sections.

Validates PR titles and descriptions to ensure they follow the Strategic PR Template
requirements with differentiator tags ([POA], [ASD], [RC], [ME], [IVL], [NSF], [FM])
and Strategic Impact sections.

Exit codes:
  0: OK (all requirements met)
  1: Missing differentiator tag in PR title
  2: Missing Strategic Impact section in PR description
  3: Invalid differentiator tag format
  4: File not found or parsing error
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


VALID_TAGS = {'POA', 'ASD', 'RC', 'ME', 'IVL', 'NSF', 'FM'}

STRATEGIC_IMPACT_FIELDS = [
    'Differentiator Tag',
    'Strategic Value',
    'Acquisition Narrative',
    'Measurable Outcomes',
    'Doctrine Alignment'
]


def validate_pr_title(title: str) -> Dict[str, Any]:
    """
    Validate PR title contains a valid differentiator tag.

    Expected format: [TAG] type(scope): description

    Args:
        title: PR title string

    Returns:
        Dict with validation results:
        {
            'valid': bool,
            'tag': str or None,
            'violations': List[str]
        }
    """
    violations = []
    tag = None

    tag_pattern = r'^\[([A-Z]+)\]\s+'
    match = re.match(tag_pattern, title)

    if not match:
        violations.append("PR title must start with a differentiator tag in format [TAG]")
        return {'valid': False, 'tag': None, 'violations': violations}

    tag = match.group(1)

    if tag not in VALID_TAGS:
        violations.append(f"Invalid differentiator tag '{tag}'. Must be one of: {', '.join(sorted(VALID_TAGS))}")
        return {'valid': False, 'tag': tag, 'violations': violations}

    remaining_title = title[len(match.group(0)):]
    commit_pattern = r'^[a-z]+(\([a-z0-9_-]+\))?: .+'

    if not re.match(commit_pattern, remaining_title):
        violations.append("PR title must follow format: [TAG] type(scope): description")
        return {'valid': False, 'tag': tag, 'violations': violations}

    return {'valid': True, 'tag': tag, 'violations': []}


def validate_strategic_impact_section(description: str) -> Dict[str, Any]:
    """
    Validate PR description contains required Strategic Impact section.

    Args:
        description: PR description markdown content

    Returns:
        Dict with validation results:
        {
            'valid': bool,
            'found_section': bool,
            'missing_fields': List[str],
            'violations': List[str]
        }
    """
    violations = []
    missing_fields = []

    strategic_impact_pattern = r'##\s*Strategic\s+Impact'
    section_match = re.search(strategic_impact_pattern, description, re.IGNORECASE)

    if section_match:
        section_start = section_match.end()
        next_section_pattern = r'\n##\s+'
        next_section_match = re.search(next_section_pattern, description[section_start:])

        if next_section_match:
            section_content = description[section_start:section_start + next_section_match.start()]
        else:
            section_content = description[section_start:]

        found_section = True
    else:
        section_content = description
        found_section = False

        has_any_field = False
        for field in STRATEGIC_IMPACT_FIELDS:
            field_pattern = rf'\*\*{re.escape(field)}\*\*:'
            if re.search(field_pattern, description, re.IGNORECASE):
                has_any_field = True
                break

        if not has_any_field:
            violations.append("PR description must include a '## Strategic Impact' section")
            return {
                'valid': False,
                'found_section': False,
                'missing_fields': STRATEGIC_IMPACT_FIELDS,
                'violations': violations
            }
        else:
            found_section = True

    for field in STRATEGIC_IMPACT_FIELDS:
        field_pattern = rf'\*\*{re.escape(field)}\*\*:'
        if not re.search(field_pattern, section_content, re.IGNORECASE):
            missing_fields.append(field)

    if missing_fields:
        violations.append(f"Strategic Impact section missing required fields: {', '.join(missing_fields)}")

    tag_selection_pattern = r'\*\*Differentiator\s+Tag\*\*:\s*\[[x✓]\]\s*\[(' + '|'.join(VALID_TAGS) + r')\]'
    if not re.search(tag_selection_pattern, section_content, re.IGNORECASE):
        violations.append("Strategic Impact section must have a selected differentiator tag checkbox")

    return {
        'valid': len(violations) == 0,
        'found_section': found_section,
        'missing_fields': missing_fields,
        'violations': violations
    }


def validate_pr_from_github_event(event_path: str) -> Dict[str, Any]:
    """
    Validate PR from GitHub Actions event payload.

    Args:
        event_path: Path to GitHub event JSON file

    Returns:
        Dict with complete validation results
    """
    try:
        with open(event_path, 'r', encoding='utf-8') as f:
            event_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {
            'valid': False,
            'violations': [f"Error reading GitHub event file: {e}"],
            'title_validation': {'valid': False, 'violations': [str(e)]},
            'description_validation': {'valid': False, 'violations': [str(e)]}
        }

    pr_data = event_data.get('pull_request', {})
    title = pr_data.get('title', '')
    description = pr_data.get('body', '') or ''

    if not title:
        return {
            'valid': False,
            'violations': ["No PR title found in event data"],
            'title_validation': {'valid': False, 'violations': ["No title found"]},
            'description_validation': {'valid': False, 'violations': ["No description found"]}
        }

    title_validation = validate_pr_title(title)
    description_validation = validate_strategic_impact_section(description)

    all_violations = title_validation['violations'] + description_validation['violations']

    return {
        'valid': title_validation['valid'] and description_validation['valid'],
        'violations': all_violations,
        'title_validation': title_validation,
        'description_validation': description_validation,
        'pr_title': title,
        'pr_number': pr_data.get('number'),
        'pr_url': pr_data.get('html_url')
    }


def validate_pr_manual(title: str, description: str) -> Dict[str, Any]:
    """
    Validate PR manually with provided title and description.

    Args:
        title: PR title
        description: PR description

    Returns:
        Dict with complete validation results
    """
    title_validation = validate_pr_title(title)
    description_validation = validate_strategic_impact_section(description)

    all_violations = title_validation['violations'] + description_validation['violations']

    return {
        'valid': title_validation['valid'] and description_validation['valid'],
        'violations': all_violations,
        'title_validation': title_validation,
        'description_validation': description_validation,
        'pr_title': title
    }


def main() -> int:
    """Main entry point for the Strategic PR validator CLI."""
    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print("  python tools/ci/strategic_pr_validator.py <github_event_path>", file=sys.stderr)
        print("  python tools/ci/strategic_pr_validator.py --manual <title> <description>", file=sys.stderr)
        return 4

    if sys.argv[1] == '--manual':
        if len(sys.argv) != 4:
            print("Manual mode requires title and description arguments", file=sys.stderr)
            return 4

        title = sys.argv[2]
        description = sys.argv[3]
        result = validate_pr_manual(title, description)
    else:
        event_path = sys.argv[1]
        result = validate_pr_from_github_event(event_path)

    if result['violations']:
        print("Strategic PR Template Violations:", file=sys.stderr)
        for violation in result['violations']:
            print(f"  - {violation}", file=sys.stderr)

        if 'pr_url' in result:
            print(f"\nPR: {result['pr_url']}", file=sys.stderr)

        print("\nRequired format:", file=sys.stderr)
        print("  Title: [TAG] type(scope): description", file=sys.stderr)
        print("  Valid tags: " + ", ".join(sorted(VALID_TAGS)), file=sys.stderr)
        print("  Description must include '## Strategic Impact' section with required fields", file=sys.stderr)

        if any('differentiator tag' in v.lower() for v in result['violations']):
            if any('invalid' in v.lower() for v in result['violations']):
                return 3  # Invalid tag format
            else:
                return 1  # Missing tag
        elif any('strategic impact' in v.lower() for v in result['violations']):
            return 2  # Missing Strategic Impact section
        else:
            return 1  # General violation

    # Success
    tag = result.get('title_validation', {}).get('tag', 'Unknown')
    print(f"Strategic PR Template validation passed: [{tag}] tag detected")
    return 0


if __name__ == '__main__':
    sys.exit(main())
