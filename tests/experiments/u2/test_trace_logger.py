"""
PHASE II — NOT USED IN PHASE I

Unit tests for experiments.u2.runtime.trace_logger module.

These tests verify:
    - TelemetryRecord dataclass and serialization
    - TraceWriter JSONL output
    - TraceReader JSONL input
    - build_telemetry_record function
    - Schema compatibility with original implementation
"""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.u2.runtime.trace_logger import (
    PHASE_II_LABEL,
    TelemetryRecord,
    TraceWriter,
    TraceReader,
    build_telemetry_record,
)


class TestPhaseLabel(unittest.TestCase):
    """Tests for the PHASE_II_LABEL constant."""

    def test_label_value(self) -> None:
        """Test that phase label matches expected value."""
        self.assertEqual(PHASE_II_LABEL, "PHASE II — NOT USED IN PHASE I")


class TestTelemetryRecord(unittest.TestCase):
    """Tests for the TelemetryRecord dataclass."""

    def test_create_basic_record(self) -> None:
        """Test creating a basic TelemetryRecord."""
        record = TelemetryRecord(
            cycle=0,
            slice="test_slice",
            mode="baseline",
            seed=42,
            item="1+1",
            result="{'outcome': 'VERIFIED'}",
            success=True,
        )
        self.assertEqual(record.cycle, 0)
        self.assertEqual(record.slice, "test_slice")
        self.assertEqual(record.mode, "baseline")
        self.assertEqual(record.seed, 42)
        self.assertEqual(record.item, "1+1")
        self.assertTrue(record.success)
        self.assertEqual(record.label, PHASE_II_LABEL)

    def test_default_label(self) -> None:
        """Test that default label is PHASE II label."""
        record = TelemetryRecord(
            cycle=0,
            slice="test",
            mode="baseline",
            seed=1,
            item="x",
            result="{}",
            success=False,
        )
        self.assertEqual(record.label, PHASE_II_LABEL)

    def test_immutability(self) -> None:
        """Test that TelemetryRecord is immutable."""
        record = TelemetryRecord(
            cycle=0,
            slice="test",
            mode="baseline",
            seed=1,
            item="x",
            result="{}",
            success=True,
        )
        with self.assertRaises(AttributeError):
            record.cycle = 1

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        record = TelemetryRecord(
            cycle=5,
            slice="arithmetic",
            mode="rfl",
            seed=12345,
            item="2+2",
            result="{'value': 4}",
            success=True,
        )
        result = record.to_dict()

        self.assertEqual(result["cycle"], 5)
        self.assertEqual(result["slice"], "arithmetic")
        self.assertEqual(result["mode"], "rfl")
        self.assertEqual(result["seed"], 12345)
        self.assertEqual(result["item"], "2+2")
        self.assertEqual(result["result"], "{'value': 4}")
        self.assertTrue(result["success"])
        self.assertEqual(result["label"], PHASE_II_LABEL)

    def test_to_json(self) -> None:
        """Test to_json serialization."""
        record = TelemetryRecord(
            cycle=0,
            slice="test",
            mode="baseline",
            seed=42,
            item="test",
            result="{}",
            success=True,
        )
        json_str = record.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["cycle"], 0)
        self.assertEqual(parsed["slice"], "test")


class TestBuildTelemetryRecord(unittest.TestCase):
    """Tests for the build_telemetry_record function."""

    def test_basic_build(self) -> None:
        """Test building a basic telemetry record."""
        record = build_telemetry_record(
            cycle=0,
            slice_name="test_slice",
            mode="baseline",
            cycle_seed=42,
            chosen_item="1+1",
            result={"outcome": "VERIFIED"},
            success=True,
        )

        self.assertEqual(record.cycle, 0)
        self.assertEqual(record.slice, "test_slice")
        self.assertEqual(record.mode, "baseline")
        self.assertEqual(record.seed, 42)
        self.assertEqual(record.item, "1+1")
        self.assertTrue(record.success)
        self.assertEqual(record.label, PHASE_II_LABEL)

    def test_result_stringified(self) -> None:
        """Test that result dict is converted to string."""
        result_dict = {"outcome": "VERIFIED", "value": 42}
        record = build_telemetry_record(
            cycle=0,
            slice_name="test",
            mode="baseline",
            cycle_seed=1,
            chosen_item="x",
            result=result_dict,
            success=True,
        )

        # Result should be string representation
        self.assertEqual(record.result, str(result_dict))

    def test_item_stringified(self) -> None:
        """Test that chosen_item is converted to string."""
        record = build_telemetry_record(
            cycle=0,
            slice_name="test",
            mode="baseline",
            cycle_seed=1,
            chosen_item=123,  # Non-string item
            result={},
            success=False,
        )

        self.assertEqual(record.item, "123")

    def test_backward_compatibility_format(self) -> None:
        """
        Test that output matches original inline implementation.

        Original code:
            telemetry_record = {
                "cycle": i,
                "slice": slice_name,
                "mode": mode,
                "seed": cycle_seed,
                "item": chosen_item,
                "result": str(mock_result),
                "success": success,
                "label": "PHASE II — NOT USED IN PHASE I",
            }
        """
        # Simulate original values
        i = 5
        slice_name = "arithmetic_simple"
        mode = "rfl"
        cycle_seed = 12345
        chosen_item = "1+2"
        mock_result = {"outcome": "VERIFIED"}
        success = True

        # Build using new function
        record = build_telemetry_record(
            cycle=i,
            slice_name=slice_name,
            mode=mode,
            cycle_seed=cycle_seed,
            chosen_item=chosen_item,
            result=mock_result,
            success=success,
        )

        # Build original format
        original = {
            "cycle": i,
            "slice": slice_name,
            "mode": mode,
            "seed": cycle_seed,
            "item": chosen_item,
            "result": str(mock_result),
            "success": success,
            "label": "PHASE II — NOT USED IN PHASE I",
        }

        # Compare
        new_dict = record.to_dict()
        self.assertEqual(new_dict, original)


class TestTraceWriter(unittest.TestCase):
    """Tests for the TraceWriter class."""

    def test_write_single_record(self) -> None:
        """Test writing a single record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"

            record = TelemetryRecord(
                cycle=0,
                slice="test",
                mode="baseline",
                seed=42,
                item="x",
                result="{}",
                success=True,
            )

            with TraceWriter(path) as writer:
                writer.write(record)

            # Verify file contents
            with open(path) as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["cycle"], 0)

    def test_write_multiple_records(self) -> None:
        """Test writing multiple records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"

            with TraceWriter(path) as writer:
                for i in range(5):
                    record = TelemetryRecord(
                        cycle=i,
                        slice="test",
                        mode="baseline",
                        seed=i * 100,
                        item=f"item_{i}",
                        result="{}",
                        success=i % 2 == 0,
                    )
                    writer.write(record)

            # Verify file contents
            with open(path) as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 5)

    def test_records_written_counter(self) -> None:
        """Test records_written property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"

            with TraceWriter(path) as writer:
                self.assertEqual(writer.records_written, 0)

                for i in range(3):
                    record = TelemetryRecord(
                        cycle=i,
                        slice="test",
                        mode="baseline",
                        seed=i,
                        item="x",
                        result="{}",
                        success=True,
                    )
                    writer.write(record)

                self.assertEqual(writer.records_written, 3)

    def test_creates_parent_directories(self) -> None:
        """Test that TraceWriter creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dirs" / "test.jsonl"

            record = TelemetryRecord(
                cycle=0,
                slice="test",
                mode="baseline",
                seed=1,
                item="x",
                result="{}",
                success=True,
            )

            with TraceWriter(path) as writer:
                writer.write(record)

            self.assertTrue(path.exists())

    def test_write_dict(self) -> None:
        """Test write_dict method for raw dictionaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"

            with TraceWriter(path) as writer:
                writer.write_dict({"custom": "data", "value": 123})

            with open(path) as f:
                line = f.readline()
                parsed = json.loads(line)

            self.assertEqual(parsed["custom"], "data")
            self.assertEqual(parsed["value"], 123)

    def test_jsonl_format(self) -> None:
        """Test that output is valid JSONL (one JSON object per line)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"

            with TraceWriter(path) as writer:
                for i in range(3):
                    record = TelemetryRecord(
                        cycle=i,
                        slice="test",
                        mode="baseline",
                        seed=i,
                        item="x",
                        result="{}",
                        success=True,
                    )
                    writer.write(record)

            # Read and parse each line separately
            with open(path) as f:
                for i, line in enumerate(f):
                    parsed = json.loads(line.strip())
                    self.assertEqual(parsed["cycle"], i)


class TestTraceReader(unittest.TestCase):
    """Tests for the TraceReader class."""

    def test_read_all(self) -> None:
        """Test reading all records from a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"

            # Write test data
            with TraceWriter(path) as writer:
                for i in range(5):
                    record = TelemetryRecord(
                        cycle=i,
                        slice=f"slice_{i}",
                        mode="baseline" if i % 2 == 0 else "rfl",
                        seed=i * 100,
                        item=f"item_{i}",
                        result=f"{{'value': {i}}}",
                        success=i % 2 == 0,
                    )
                    writer.write(record)

            # Read back
            reader = TraceReader(path)
            records = reader.read_all()

            self.assertEqual(len(records), 5)
            for i, record in enumerate(records):
                self.assertEqual(record.cycle, i)
                self.assertEqual(record.slice, f"slice_{i}")

    def test_read_dicts(self) -> None:
        """Test reading records as raw dictionaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"

            with TraceWriter(path) as writer:
                writer.write_dict({"custom_field": "value1"})
                writer.write_dict({"custom_field": "value2"})

            reader = TraceReader(path)
            dicts = reader.read_dicts()

            self.assertEqual(len(dicts), 2)
            self.assertEqual(dicts[0]["custom_field"], "value1")
            self.assertEqual(dicts[1]["custom_field"], "value2")

    def test_roundtrip(self) -> None:
        """Test that write/read roundtrip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"

            original_records = [
                TelemetryRecord(
                    cycle=i,
                    slice="test_slice",
                    mode="baseline",
                    seed=42 + i,
                    item=f"item_{i}",
                    result=f"{{'outcome': 'VERIFIED', 'i': {i}}}",
                    success=True,
                )
                for i in range(10)
            ]

            with TraceWriter(path) as writer:
                for record in original_records:
                    writer.write(record)

            reader = TraceReader(path)
            read_records = reader.read_all()

            self.assertEqual(len(read_records), len(original_records))
            for orig, read in zip(original_records, read_records):
                self.assertEqual(orig.cycle, read.cycle)
                self.assertEqual(orig.slice, read.slice)
                self.assertEqual(orig.mode, read.mode)
                self.assertEqual(orig.seed, read.seed)
                self.assertEqual(orig.item, read.item)
                self.assertEqual(orig.result, read.result)
                self.assertEqual(orig.success, read.success)
                self.assertEqual(orig.label, read.label)


class TestSchemaCompatibility(unittest.TestCase):
    """
    Tests ensuring schema compatibility with original JSONL format.

    PHASE II — NOT USED IN PHASE I
    """

    def test_field_names_match_original(self) -> None:
        """Test that field names exactly match original schema."""
        record = TelemetryRecord(
            cycle=0,
            slice="test",
            mode="baseline",
            seed=42,
            item="x",
            result="{}",
            success=True,
        )

        expected_keys = {"cycle", "slice", "mode", "seed", "item", "result", "success", "label"}
        actual_keys = set(record.to_dict().keys())

        self.assertEqual(actual_keys, expected_keys)

    def test_json_output_byte_compatible(self) -> None:
        """Test that JSON output format is compatible with original."""
        record = build_telemetry_record(
            cycle=0,
            slice_name="arithmetic_simple",
            mode="baseline",
            cycle_seed=1608637542,
            chosen_item="1+1",
            result={"outcome": "VERIFIED"},
            success=True,
        )

        json_str = record.to_json()

        # Should be parseable
        parsed = json.loads(json_str)

        # Original format expectations
        self.assertIsInstance(parsed["cycle"], int)
        self.assertIsInstance(parsed["slice"], str)
        self.assertIsInstance(parsed["mode"], str)
        self.assertIsInstance(parsed["seed"], int)
        self.assertIsInstance(parsed["item"], str)
        self.assertIsInstance(parsed["result"], str)
        self.assertIsInstance(parsed["success"], bool)
        self.assertIsInstance(parsed["label"], str)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

