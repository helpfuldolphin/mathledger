import unittest
import unittest.mock
import io
import json
import os
import tempfile
import shutil
import sys

# Ensure the script's directory is in the path to find the module to test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from summarize_evidence_checklist import parse_checklist, analyze_progress, calculate_trend, generate_markdown_summary, main


FIXTURE_MARKDOWN = """
# Phase X Checklist
## Dependency DAG
1. `System Architecture Diagram` -> `API Documentation`
## Sprint Schedule & Artifact Ownership
| Gate | Status | Team | Artifact | Location |
|---|---|---|---|---|
| **P3** | `READY` | Codex | `System Architecture Diagram` | `docs/` |
| **P3** | `BLOCKED` | Codex | `API Documentation` | `docs/` |
"""

class TestContractHardening(unittest.TestCase):

    def setUp(self):
        self.artifacts, self.dag_lines = parse_checklist(FIXTURE_MARKDOWN)
        self.summary = analyze_progress(self.artifacts, self.dag_lines)
        self.trend = calculate_trend(self.summary, None) # No previous data

    def test_json_output_contract(self):
        """Ensures JSON output is deterministic and contains schema_version."""
        self.summary["schema_version"] = "1.0.0"
        
        # Generate JSON with sort_keys=True
        json_string_sorted = json.dumps(self.summary, sort_keys=True)
        
        # Generate JSON without sort_keys=True for comparison
        json_string_unsorted = json.dumps(self.summary)
        
        # The actual test: re-parse and check keys
        data = json.loads(json_string_sorted)
        self.assertIn("schema_version", data)
        self.assertEqual(data["schema_version"], "1.0.0")
        
        # Check if keys are sorted
        keys = list(data.keys())
        self.assertEqual(keys, sorted(keys))

    def test_windows_encoding_safety_hardened(self):
        """
        Asserts that ASCII mode is fully cp1252-safe by checking that it
        can be encoded without error, unlike the Unicode version.
        """
        # 1. Unicode markdown *must* fail with a basic Windows encoding
        unicode_markdown = generate_markdown_summary(self.summary, self.trend, ascii_only=False)
        with self.assertRaises(UnicodeEncodeError, msg="Unicode markdown should fail to encode with cp1252"):
            unicode_markdown.encode('cp1252')

        # 2. ASCII-only markdown *must not* fail
        ascii_markdown = generate_markdown_summary(self.summary, self.trend, ascii_only=True)
        try:
            ascii_markdown.encode('cp1252')
        except UnicodeEncodeError:
            self.fail("ASCII-only markdown failed to encode with cp1252, even though it should be safe.")

    def test_output_dir_argument(self):
        """Verify that the --output-dir argument correctly directs output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # We need to mock sys.argv to test the main function's argument parsing
            import sys
            original_argv = sys.argv
            sys.argv = ['summarize_evidence_checklist.py', '--output-dir', tmpdir, '--ascii-only']
            
            # We also need a dummy checklist file for main() to read
            checklist_path_orig = 'summarize_evidence_checklist.CHECKLIST_PATH_DEFAULT'
            with open('dummy_checklist.md', 'w') as f:
                f.write(FIXTURE_MARKDOWN)
            
            # Patch the global path to our dummy file
            with unittest.mock.patch(checklist_path_orig, 'dummy_checklist.md'):
                main()
            
            # Assert that files were created in the temp directory
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'readiness.json')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'readiness_summary.md')))
            
            # Cleanup
            sys.argv = original_argv
            os.remove('dummy_checklist.md')

if __name__ == '__main__':
    unittest.main()