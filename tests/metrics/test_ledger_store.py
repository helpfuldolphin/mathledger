"""
PHASE II — NOT USED IN PHASE I
Tests for the MetricLedgerStore abstraction and its implementations.
"""
import unittest
import os
import json
import shutil
import sys

sys.path.insert(0, '.')
from experiments.metric_consistency_auditor import (
    MetricLedgerStore,
    InMemoryLedgerStore,
    FilesystemLedgerStore,
)

class TestLedgerStore(unittest.TestCase):

    def setUp(self):
        self.mock_ledger = {
            "ledger_id": "test_id_123",
            "data": "some_value"
        }
        self.test_dir = "test_artifacts"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_in_memory_store(self):
        """Tests the InMemoryLedgerStore save and get methods."""
        store = InMemoryLedgerStore()
        
        # Test save
        store.save(self.mock_ledger)
        self.assertIn("test_id_123", store._store)
        
        # Test get
        retrieved = store.get("test_id_123")
        self.assertEqual(self.mock_ledger, retrieved)
        
        # Test get non-existent
        retrieved_none = store.get("non_existent_id")
        self.assertIsNone(retrieved_none)

    def test_filesystem_store(self):
        """Tests the FilesystemLedgerStore save method."""
        store = FilesystemLedgerStore(base_path=self.test_dir)
        
        store.save(self.mock_ledger)
        
        # Verify file was created
        expected_path = os.path.join(self.test_dir, "test_id_123.json")
        self.assertTrue(os.path.exists(expected_path))
        
        # Verify file content
        with open(expected_path, 'r') as f:
            content = json.load(f)
        self.assertEqual(self.mock_ledger, content)

    def test_abc_enforcement(self):
        """Ensures the ABC prevents instantiation without implementation."""
        with self.assertRaises(TypeError):
            class BadStore(MetricLedgerStore):
                pass
            b = BadStore()

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
