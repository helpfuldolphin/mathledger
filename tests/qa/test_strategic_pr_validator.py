import unittest
import tempfile
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ci"))

from strategic_pr_validator import (
    validate_pr_title,
    validate_strategic_impact_section,
    validate_pr_from_github_event,
    validate_pr_manual,
    VALID_TAGS
)


class TestStrategicPRValidator(unittest.TestCase):

    def test_valid_pr_title(self):
        """Test valid PR title with differentiator tag."""
        title = "[POA] feat(derive): implement guided derivation with 95% success rate"
        result = validate_pr_title(title)

        self.assertTrue(result['valid'])
        self.assertEqual(result['tag'], 'POA')
        self.assertEqual(result['violations'], [])

    def test_invalid_pr_title_no_tag(self):
        """Test PR title without differentiator tag."""
        title = "feat(derive): implement guided derivation"
        result = validate_pr_title(title)

        self.assertFalse(result['valid'])
        self.assertIsNone(result['tag'])
        self.assertIn("must start with a differentiator tag", result['violations'][0])

    def test_invalid_pr_title_bad_tag(self):
        """Test PR title with invalid differentiator tag."""
        title = "[INVALID] feat(derive): implement guided derivation"
        result = validate_pr_title(title)

        self.assertFalse(result['valid'])
        self.assertEqual(result['tag'], 'INVALID')
        self.assertIn("Invalid differentiator tag", result['violations'][0])

    def test_invalid_pr_title_bad_format(self):
        """Test PR title with bad conventional commit format."""
        title = "[POA] bad format here"
        result = validate_pr_title(title)

        self.assertFalse(result['valid'])
        self.assertEqual(result['tag'], 'POA')
        self.assertIn("must follow format", result['violations'][0])

    def test_valid_strategic_impact_section(self):
        """Test valid Strategic Impact section."""
        description = """


**Differentiator Tag**: [x] [POA] [ ] [ASD] [ ] [RC] [ ] [ME] [ ] [IVL] [ ] [NSF] [ ] [FM]

**Strategic Value**: Demonstrates autonomous theorem proving capabilities

**Acquisition Narrative**: Advances competitive positioning in automated reasoning

**Measurable Outcomes**: 95% success rate improvement

**Doctrine Alignment**: Automated reasoning systems that scale beyond human capacity


More content here.
"""
        result = validate_strategic_impact_section(description)

        self.assertTrue(result['valid'])
        self.assertTrue(result['found_section'])
        self.assertEqual(result['missing_fields'], [])
        self.assertEqual(result['violations'], [])

    def test_missing_strategic_impact_section(self):
        """Test PR description without Strategic Impact section."""
        description = """


This is a regular PR description without strategic impact.


- Added feature X
- Fixed bug Y
"""
        result = validate_strategic_impact_section(description)

        self.assertFalse(result['valid'])
        self.assertFalse(result['found_section'])
        self.assertIn("must include a '## Strategic Impact' section", result['violations'][0])

    def test_incomplete_strategic_impact_section(self):
        """Test Strategic Impact section missing required fields."""
        description = """

**Differentiator Tag**: [x] [POA]

**Strategic Value**: Some value

"""
        result = validate_strategic_impact_section(description)

        self.assertFalse(result['valid'])
        self.assertTrue(result['found_section'])
        self.assertGreater(len(result['missing_fields']), 0)
        self.assertIn("missing required fields", result['violations'][0])

    def test_github_event_validation(self):
        """Test validation from GitHub event payload."""
        event_data = {
            "pull_request": {
                "number": 123,
                "title": "[RC] test(integration): achieve 90% test coverage",
                "body": """## Strategic Impact

**Differentiator Tag**: [x] [RC] [ ] [POA] [ ] [ASD] [ ] [ME] [ ] [IVL] [ ] [NSF] [ ] [FM]

**Strategic Value**: Enhances system reliability and correctness verification

**Acquisition Narrative**: Demonstrates mission-critical reliability for production deployment

**Measurable Outcomes**: 90% test coverage achieved for core derivation engine

**Doctrine Alignment**: Mission-critical reliability for production deployment
""",
                "html_url": "https://github.com/test/repo/pull/123"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(event_data, f)
            event_path = f.name

        try:
            result = validate_pr_from_github_event(event_path)

            self.assertTrue(result['valid'])
            self.assertEqual(result['violations'], [])
            self.assertEqual(result['pr_number'], 123)
            self.assertTrue(result['title_validation']['valid'])
            self.assertTrue(result['description_validation']['valid'])
        finally:
            os.unlink(event_path)

    def test_manual_validation(self):
        """Test manual validation with title and description."""
        title = "[ME] feat(metrics): add v1 schema supporting 10K+ proofs/hour monitoring"
        description = """## Strategic Impact

**Differentiator Tag**: [x] [ME] [ ] [POA] [ ] [ASD] [ ] [RC] [ ] [IVL] [ ] [NSF] [ ] [FM]

**Strategic Value**: Provides measurable evidence of system performance

**Acquisition Narrative**: Quantifiable performance metrics for stakeholder confidence

**Measurable Outcomes**: Support for 10K+ proofs/hour monitoring capability

**Doctrine Alignment**: Quantifiable performance metrics for stakeholder confidence
"""

        result = validate_pr_manual(title, description)

        self.assertTrue(result['valid'])
        self.assertEqual(result['violations'], [])
        self.assertEqual(result['title_validation']['tag'], 'ME')

    def test_all_valid_tags(self):
        """Test that all defined valid tags are accepted."""
        for tag in VALID_TAGS:
            title = f"[{tag}] feat(test): test {tag} tag"
            result = validate_pr_title(title)

            self.assertTrue(result['valid'], f"Tag {tag} should be valid")
            self.assertEqual(result['tag'], tag)


if __name__ == '__main__':
    unittest.main()
