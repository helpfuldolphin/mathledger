import unittest
import tempfile
import os
from phase_ix_progress import parse_progress

class TestProgressParser(unittest.TestCase):

    def test_parse_progress(self):
        """
        Tests that the parser correctly counts TODO, IN_PROGRESS, and DONE items.
        """
        markdown_fixture = """
# Project Plan

- [ ] Task 1 (TODO)
- [x] Task 2 (DONE)
- [ ] Task 3 (TODO)
- [~] Task 4 (IN_PROGRESS)
- [X] Task 5 (DONE, capital X)
- [ ] Another item to do.

## Sub-section
- [~] An in-progress task.
"""
        # Create a temporary file to write the fixture to
        # 'delete=False' is needed on Windows to allow reading the file after writing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.md') as tmp:
            tmp.write(markdown_fixture)
            tmp_path = tmp.name

        try:
            # Run the parser on the temporary file
            result = parse_progress(tmp_path)
            
            # Define expected counts
            expected = {
                "TODO": 3,
                "IN_PROGRESS": 2,
                "DONE": 2
            }
            
            # Assert that the result matches the expected counts
            self.assertEqual(result, expected)
        finally:
            # Clean up the temporary file
            os.remove(tmp_path)

    def test_file_not_found(self):
        """
        Tests that the parser returns an error if the file does not exist.
        """
        non_existent_file = "non_existent_file.md"
        result = parse_progress(non_existent_file)
        self.assertIn("error", result)
        self.assertIn("File not found", result["error"])

if __name__ == '__main__':
    unittest.main()
