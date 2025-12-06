#!/usr/bin/env python3
"""
Unit tests for the branch guard script.

Tests branch naming validation with mocked git subprocess calls.
"""

import unittest
from unittest.mock import patch, MagicMock
import subprocess
import sys
import os

# Add the tools directory to the path so we can import branch_guard
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools', 'ci-local'))

from branch_guard import validate_branch_name, get_current_branch, main


class TestBranchGuard(unittest.TestCase):
    """Test cases for branch guard functionality."""

    def test_validate_branch_name_valid_cases(self):
        """Test valid branch names that should pass validation."""
        valid_branches = [
            'feature/user-authentication',
            'feature/api-endpoints',
            'perf/database-optimization',
            'perf/query-caching',
            'ops/deployment-scripts',
            'ops/monitoring-setup',
            'qa/test-coverage',
            'qa/integration-tests',
            'devxp/local-guardrails',
            'devxp/development-tools',
            'docs/api-documentation',
            'docs/user-guide',
            'feature/ab',
            'perf/a1b2c3',
            'ops/test-123',
            'qa/456-test',
            'devxp/abc-def-ghi',
            'docs/123-456-789'
        ]

        for branch in valid_branches:
            with self.subTest(branch=branch):
                is_valid, error_msg = validate_branch_name(branch)
                self.assertTrue(is_valid, f"Branch '{branch}' should be valid: {error_msg}")
                self.assertEqual(error_msg, "")

    def test_validate_branch_name_invalid_cases(self):
        """Test invalid branch names that should fail validation."""
        invalid_cases = [
            # Missing prefix
            ('user-auth', 'missing prefix'),
            ('api-endpoints', 'missing prefix'),
            ('test-123', 'missing prefix'),

            # Invalid prefix
            ('bugfix/user-auth', 'invalid prefix'),
            ('hotfix/urgent-fix', 'invalid prefix'),
            ('release/v1.0', 'invalid prefix'),
            ('chore/cleanup', 'invalid prefix'),

            # Invalid characters
            ('feature/User-Auth', 'uppercase not allowed'),
            ('feature/user_auth', 'underscore not allowed'),
            ('feature/user.auth', 'dot not allowed'),
            ('feature/user@auth', 'special char not allowed'),
            ('feature/user auth', 'space not allowed'),

            # Empty or malformed
            ('feature/', 'empty suffix'),
            ('feature/-test', 'dash at start'),
            ('feature/--test', 'double dash'),
            ('feature/-', 'only dash'),
            ('', 'empty string'),

            # Wrong format
            ('feature', 'no slash'),
            ('feature/user/auth', 'multiple slashes'),
            ('/feature/user-auth', 'slash at start'),
        ]

        for branch, description in invalid_cases:
            with self.subTest(branch=branch, description=description):
                is_valid, error_msg = validate_branch_name(branch)
                self.assertFalse(is_valid, f"Branch '{branch}' should be invalid ({description})")
                if branch == '':
                    self.assertIn("Empty branch name", error_msg)
                else:
                    self.assertIn("does not match required pattern", error_msg)

    def test_validate_branch_name_edge_cases(self):
        """Test edge cases for branch name validation."""
        # Single character after prefix (should be invalid per regex)
        is_valid, error_msg = validate_branch_name('feature/a')
        self.assertFalse(is_valid, f"Single character should be invalid: {error_msg}")

        # Two characters (minimum valid)
        is_valid, error_msg = validate_branch_name('feature/ab')
        self.assertTrue(is_valid, f"Two characters should be valid: {error_msg}")

        # Very long name
        long_name = 'feature/' + 'a' * 100
        is_valid, error_msg = validate_branch_name(long_name)
        self.assertTrue(is_valid, f"Long name should be valid: {error_msg}")

    @patch('subprocess.run')
    def test_get_current_branch_success(self, mock_run):
        """Test successful git branch retrieval."""
        # Mock successful git command
        mock_result = MagicMock()
        mock_result.stdout = 'feature/test-branch\n'
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        branch = get_current_branch()
        self.assertEqual(branch, 'feature/test-branch')

        # Verify git command was called correctly
        mock_run.assert_called_once_with(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )

    @patch('subprocess.run')
    def test_get_current_branch_git_error(self, mock_run):
        """Test git command failure."""
        # Mock git command failure
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git', 'Git error')

        branch = get_current_branch()
        self.assertIsNone(branch)

    @patch('subprocess.run')
    def test_get_current_branch_git_not_found(self, mock_run):
        """Test when git command is not found."""
        # Mock FileNotFoundError
        mock_run.side_effect = FileNotFoundError()

        branch = get_current_branch()
        self.assertIsNone(branch)

    @patch('branch_guard.get_current_branch')
    def test_main_success(self, mock_get_branch):
        """Test main function with valid branch."""
        mock_get_branch.return_value = 'feature/test-branch'

        result = main()
        self.assertEqual(result, 0)

    @patch('branch_guard.get_current_branch')
    def test_main_invalid_branch(self, mock_get_branch):
        """Test main function with invalid branch."""
        mock_get_branch.return_value = 'invalid-branch'

        result = main()
        self.assertEqual(result, 1)

    @patch('branch_guard.get_current_branch')
    def test_main_git_error(self, mock_get_branch):
        """Test main function when git fails."""
        mock_get_branch.return_value = None

        result = main()
        self.assertEqual(result, 1)

    @patch('branch_guard.get_current_branch')
    def test_main_main_branch(self, mock_get_branch):
        """Test main function with main branch (should warn but pass)."""
        mock_get_branch.return_value = 'main'

        result = main()
        self.assertEqual(result, 0)

    def test_regex_pattern_comprehensive(self):
        """Comprehensive test of the regex pattern."""
        # Test all valid prefixes
        prefixes = ['feature', 'perf', 'ops', 'qa', 'devxp', 'docs']
        suffixes = ['ab', 'a1', '1a', 'a-b', 'a1b2', 'a-b-c', '123-456']

        for prefix in prefixes:
            for suffix in suffixes:
                branch = f"{prefix}/{suffix}"
                with self.subTest(branch=branch):
                    is_valid, error_msg = validate_branch_name(branch)
                    self.assertTrue(is_valid, f"Branch '{branch}' should be valid: {error_msg}")

    def test_error_message_content(self):
        """Test that error messages contain helpful information."""
        is_valid, error_msg = validate_branch_name('invalid-branch')

        self.assertFalse(is_valid)
        self.assertIn("does not match required pattern", error_msg)
        self.assertIn("Expected:", error_msg)
        self.assertIn("Examples:", error_msg)
        self.assertIn("feature/user-auth", error_msg)


if __name__ == '__main__':
    unittest.main()
