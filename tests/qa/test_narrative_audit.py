import unittest
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ci"))

from narrative_audit import (
    extract_tag_from_title,
    analyze_pr_compliance,
    generate_narrative_audit_report,
    VALID_TAGS,
    TAG_DESCRIPTIONS
)


class TestNarrativeAudit(unittest.TestCase):

    def test_extract_tag_from_title(self):
        """Test extracting differentiator tags from PR titles."""
        test_cases = [
            ("[POA] feat(derive): implement guided derivation", "POA"),
            ("[ASD] perf(canon): optimize normalization", "ASD"),
            ("[RC] test(integration): add smoke tests", "RC"),
            ("feat(derive): no tag here", None),
            ("[INVALID] feat(test): invalid tag", None),
            ("[POA] [ASD] feat(test): multiple tags", "POA"),  # Should get first one
        ]

        for title, expected_tag in test_cases:
            with self.subTest(title=title):
                result = extract_tag_from_title(title)
                self.assertEqual(result, expected_tag)

    def test_analyze_pr_compliance_empty(self):
        """Test PR compliance analysis with no PRs."""
        prs = []
        result = analyze_pr_compliance(prs)

        self.assertEqual(result['total_prs'], 0)
        self.assertEqual(result['compliant_prs'], 0)
        self.assertEqual(result['non_compliant_prs'], 0)
        self.assertEqual(result['compliance_rate'], 0)
        self.assertEqual(result['tag_counts'], {})

    def test_analyze_pr_compliance_mixed(self):
        """Test PR compliance analysis with mixed compliant/non-compliant PRs."""
        prs = [
            {
                'number': 1,
                'title': '[POA] feat(derive): implement guided derivation',
                'author': 'alice',
                'merged_at': '2024-01-01'
            },
            {
                'number': 2,
                'title': 'feat(api): add new endpoint',  # No tag
                'author': 'bob',
                'merged_at': '2024-01-02'
            },
            {
                'number': 3,
                'title': '[RC] test(integration): add smoke tests',
                'author': 'alice',
                'merged_at': '2024-01-03'
            },
            {
                'number': 4,
                'title': '[POA] perf(worker): optimize queue processing',
                'author': 'charlie',
                'merged_at': '2024-01-04'
            }
        ]

        result = analyze_pr_compliance(prs)

        self.assertEqual(result['total_prs'], 4)
        self.assertEqual(result['compliant_prs'], 3)
        self.assertEqual(result['non_compliant_prs'], 1)
        self.assertEqual(result['compliance_rate'], 75.0)
        self.assertEqual(result['tag_counts'], {'POA': 2, 'RC': 1})
        self.assertEqual(result['tag_by_author']['alice'], ['POA', 'RC'])
        self.assertEqual(result['tag_by_author']['charlie'], ['POA'])
        self.assertNotIn('bob', result['tag_by_author'])

    def test_generate_narrative_audit_report(self):
        """Test narrative audit report generation."""
        analysis = {
            'total_prs': 5,
            'compliant_prs': 4,
            'non_compliant_prs': 1,
            'compliance_rate': 80.0,
            'tag_counts': {'POA': 2, 'RC': 1, 'ME': 1},
            'tag_by_author': {
                'alice': ['POA', 'RC'],
                'bob': ['ME'],
                'charlie': ['POA']
            },
            'compliant_pr_list': [
                {'number': 1, 'title': '[POA] feat(derive): test', 'author': 'alice', 'tag': 'POA'},
                {'number': 2, 'title': '[RC] test(integration): test', 'author': 'alice', 'tag': 'RC'},
                {'number': 3, 'title': '[ME] feat(metrics): test', 'author': 'bob', 'tag': 'ME'},
                {'number': 4, 'title': '[POA] perf(worker): test', 'author': 'charlie', 'tag': 'POA'}
            ],
            'non_compliant_pr_list': [
                {'number': 5, 'title': 'feat(api): no tag', 'author': 'dave', 'tag': None}
            ]
        }

        report = generate_narrative_audit_report(analysis, days=7)

        self.assertIn("Strategic PR Differentiator Audit Report", report)
        self.assertIn("Compliance Rate**: 80.0%", report)
        self.assertIn("Total PRs Analyzed**: 5", report)
        self.assertIn("Strategic PR Differentiator Audit Report", report)
        self.assertIn("Top Contributors", report)
        self.assertIn("Non-Compliant PRs", report)
        self.assertIn("Recent Compliant PRs", report)
        self.assertIn("Recommendations", report)

        for tag in VALID_TAGS:
            self.assertIn(f"[{tag}]", report)
            self.assertIn(TAG_DESCRIPTIONS[tag], report)

        self.assertIn("alice**: 2 PRs", report)
        self.assertIn("PR #5: feat(api): no tag", report)
        self.assertIn("[POA] PR #1:", report)

    def test_generate_report_perfect_compliance(self):
        """Test report generation with 100% compliance."""
        analysis = {
            'total_prs': 3,
            'compliant_prs': 3,
            'non_compliant_prs': 0,
            'compliance_rate': 100.0,
            'tag_counts': {'POA': 1, 'RC': 1, 'ME': 1},
            'tag_by_author': {'alice': ['POA', 'RC', 'ME']},
            'compliant_pr_list': [
                {'number': 1, 'title': '[POA] feat(derive): test', 'author': 'alice', 'tag': 'POA'},
                {'number': 2, 'title': '[RC] test(integration): test', 'author': 'alice', 'tag': 'RC'},
                {'number': 3, 'title': '[ME] feat(metrics): test', 'author': 'alice', 'tag': 'ME'}
            ],
            'non_compliant_pr_list': []
        }

        report = generate_narrative_audit_report(analysis)

        self.assertIn("Compliance Rate**: 100.0%", report)
        self.assertIn("Excellent Progress", report)
        self.assertNotIn("Non-Compliant PRs", report)

    def test_generate_report_low_compliance(self):
        """Test report generation with low compliance."""
        analysis = {
            'total_prs': 10,
            'compliant_prs': 3,
            'non_compliant_prs': 7,
            'compliance_rate': 30.0,
            'tag_counts': {'POA': 2, 'RC': 1},
            'tag_by_author': {'alice': ['POA', 'RC', 'POA']},
            'compliant_pr_list': [
                {'number': 1, 'title': '[POA] feat(derive): test', 'author': 'alice', 'tag': 'POA'}
            ],
            'non_compliant_pr_list': [
                {'number': i, 'title': f'feat(test): no tag {i}', 'author': 'bob', 'tag': None}
                for i in range(2, 9)
            ]
        }

        report = generate_narrative_audit_report(analysis)

        self.assertIn("Compliance Rate**: 30.0%", report)
        self.assertIn("Action Required", report)
        self.assertIn("Needs Attention", report)

        unused_tags = VALID_TAGS - {'POA', 'RC'}
        for tag in unused_tags:
            self.assertIn(tag, report)


if __name__ == '__main__':
    unittest.main()
