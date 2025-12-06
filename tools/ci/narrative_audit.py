#!/usr/bin/env python3
"""
Weekly Narrative Audit for Strategic PR Differentiator adoption tracking.

Analyzes merged PRs to generate adoption reports for strategic differentiator tags
([POA], [ASD], [RC], [ME], [IVL], [NSF], [FM]) and acquisition narrative progress.

Exit codes:
  0: OK (report generated successfully)
  1: Error generating report
"""

import re
import sys
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter


VALID_TAGS = {'POA', 'ASD', 'RC', 'ME', 'IVL', 'NSF', 'FM'}

TAG_DESCRIPTIONS = {
    'POA': 'Proof of Automation',
    'ASD': 'Algorithmic Superiority Demonstration',
    'RC': 'Reliability & Correctness',
    'ME': 'Metrics & Evidence',
    'IVL': 'Integration & Validation Layer',
    'NSF': 'Network Security & Forensics',
    'FM': 'Formal Methods'
}


def extract_tag_from_title(title: str) -> Optional[str]:
    """Extract differentiator tag from PR title."""
    tag_pattern = r'^\[([A-Z]+)\]\s+'
    match = re.match(tag_pattern, title)
    if match:
        tag = match.group(1)
        return tag if tag in VALID_TAGS else None
    return None


def get_merged_prs_since(days: int = 7, base_branch: str = "integrate/ledger-v0.1") -> List[Dict[str, Any]]:
    """
    Get merged PRs from the last N days using git log.

    Args:
        days: Number of days to look back
        base_branch: Base branch to analyze

    Returns:
        List of PR data dictionaries
    """
    since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    try:
        cmd = [
            'git', 'log',
            f'--since={since_date}',
            '--merges',
            '--pretty=format:%H|%s|%ai|%an',
            base_branch
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        prs = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = line.split('|', 3)
            if len(parts) != 4:
                continue

            commit_hash, subject, date, author = parts

            pr_match = re.search(r'Merge pull request #(\d+)', subject)
            if not pr_match:
                continue

            pr_number = int(pr_match.group(1))

            title_match = re.search(r'Merge pull request #\d+ from [^/]+/[^\s]+\s*(.+)', subject)
            pr_title = title_match.group(1).strip() if title_match else subject

            prs.append({
                'number': pr_number,
                'title': pr_title,
                'merged_at': date,
                'author': author,
                'commit_hash': commit_hash
            })

        return prs

    except subprocess.CalledProcessError as e:
        print(f"Error getting merged PRs: {e}", file=sys.stderr)
        return []


def analyze_pr_compliance(prs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze PR compliance with Strategic PR Template requirements.

    Args:
        prs: List of PR data dictionaries

    Returns:
        Analysis results dictionary
    """
    total_prs = len(prs)
    compliant_prs = []
    non_compliant_prs = []
    tag_counts = Counter()
    tag_by_author = defaultdict(list)

    for pr in prs:
        tag = extract_tag_from_title(pr['title'])

        if tag:
            compliant_prs.append(pr)
            tag_counts[tag] += 1
            tag_by_author[pr['author']].append(tag)
            pr['tag'] = tag
        else:
            non_compliant_prs.append(pr)
            pr['tag'] = None

    compliance_rate = (len(compliant_prs) / total_prs * 100) if total_prs > 0 else 0

    return {
        'total_prs': total_prs,
        'compliant_prs': len(compliant_prs),
        'non_compliant_prs': len(non_compliant_prs),
        'compliance_rate': compliance_rate,
        'tag_counts': dict(tag_counts),
        'tag_by_author': dict(tag_by_author),
        'compliant_pr_list': compliant_prs,
        'non_compliant_pr_list': non_compliant_prs
    }


def generate_narrative_audit_report(analysis: Dict[str, Any], days: int = 7) -> str:
    """
    Generate narrative audit report in markdown format.

    Args:
        analysis: Analysis results from analyze_pr_compliance
        days: Number of days analyzed

    Returns:
        Markdown report string
    """
    report_date = datetime.now().strftime('%Y-%m-%d')
    period = f"Last {days} days"

    report = f"""# Strategic PR Differentiator Audit Report

**Report Date**: {report_date}
**Analysis Period**: {period}
**Total PRs Analyzed**: {analysis['total_prs']}


- **Compliance Rate**: {analysis['compliance_rate']:.1f}% ({analysis['compliant_prs']}/{analysis['total_prs']} PRs)
- **Strategic Differentiator Coverage**: {len(analysis['tag_counts'])}/{len(VALID_TAGS)} tags used
- **Acquisition Narrative Progress**: {'On Track' if analysis['compliance_rate'] >= 90 else 'Needs Attention'}


"""

    report += "| Tag | Description | Count | Percentage |\n"
    report += "|-----|-------------|-------|------------|\n"

    for tag in sorted(VALID_TAGS):
        count = analysis['tag_counts'].get(tag, 0)
        percentage = (count / analysis['total_prs'] * 100) if analysis['total_prs'] > 0 else 0
        description = TAG_DESCRIPTIONS[tag]
        report += f"| [{tag}] | {description} | {count} | {percentage:.1f}% |\n"

    unused_tags = VALID_TAGS - set(analysis['tag_counts'].keys())
    if unused_tags:
        report += f"\n**Unused Tags**: {', '.join(sorted(unused_tags))}\n"

    if analysis['tag_by_author']:
        report += "\n## Top Contributors\n\n"
        author_counts = {author: len(tags) for author, tags in analysis['tag_by_author'].items()}
        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        for author, count in top_authors:
            tags = analysis['tag_by_author'][author]
            tag_summary = ', '.join(f"[{tag}]" for tag in sorted(set(tags)))
            report += f"- **{author}**: {count} PRs ({tag_summary})\n"

    if analysis['non_compliant_pr_list']:
        report += f"\n## Non-Compliant PRs ({len(analysis['non_compliant_pr_list'])})\n\n"
        for pr in analysis['non_compliant_pr_list']:
            report += f"- PR #{pr['number']}: {pr['title']} (by {pr['author']})\n"

    if analysis['compliant_pr_list']:
        report += f"\n## Recent Compliant PRs ({len(analysis['compliant_pr_list'])})\n\n"
        for pr in analysis['compliant_pr_list'][:10]:  # Show up to 10 recent ones
            report += f"- [{pr['tag']}] PR #{pr['number']}: {pr['title']} (by {pr['author']})\n"

    report += "\n## Recommendations\n\n"

    if analysis['compliance_rate'] < 100:
        report += f"- **Immediate Action**: {len(analysis['non_compliant_pr_list'])} PRs need strategic differentiator tags\n"

    if unused_tags:
        report += f"- **Coverage Gap**: Consider PRs for unused tags: {', '.join(sorted(unused_tags))}\n"

    if analysis['compliance_rate'] >= 90:
        report += "- **Excellent Progress**: Maintain current compliance rate\n"
    elif analysis['compliance_rate'] >= 70:
        report += "- **Good Progress**: Focus on reaching 100% compliance\n"
    else:
        report += "- **Action Required**: Significant improvement needed to reach sprint goal\n"

    report += "\n## Next Steps\n\n"
    report += "1. Review non-compliant PRs and add appropriate differentiator tags\n"
    report += "2. Ensure all future PRs include Strategic Impact sections\n"
    report += "3. Focus on underrepresented differentiator categories\n"
    report += "4. Continue building acquisition narrative coherence\n"

    report += f"\n---\n*Generated by Strategic PR Audit System on {report_date}*\n"

    return report


def main() -> int:
    """Main entry point for the narrative audit CLI."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate Strategic PR Differentiator audit report')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze (default: 7)')
    parser.add_argument('--base-branch', default='integrate/ledger-v0.1', help='Base branch to analyze')
    parser.add_argument('--output', help='Output file path (default: stdout)')
    parser.add_argument('--format', choices=['markdown', 'json'], default='markdown', help='Output format')

    args = parser.parse_args()

    try:
        prs = get_merged_prs_since(args.days, args.base_branch)

        if not prs:
            print(f"No merged PRs found in the last {args.days} days", file=sys.stderr)
            analysis = {
                'total_prs': 0,
                'compliant_prs': 0,
                'non_compliant_prs': 0,
                'compliance_rate': 0,
                'tag_counts': {},
                'tag_by_author': {},
                'compliant_pr_list': [],
                'non_compliant_pr_list': []
            }
        else:
            analysis = analyze_pr_compliance(prs)

        if args.format == 'json':
            output = json.dumps(analysis, indent=2, default=str)
        else:
            output = generate_narrative_audit_report(analysis, args.days)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Audit report written to {args.output}")
        else:
            print(output)

        return 0

    except Exception as e:
        print(f"Error generating audit report: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
