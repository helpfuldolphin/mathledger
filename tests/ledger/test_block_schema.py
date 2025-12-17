"""Tests for Block Schema Contract.

Tests the schema validation for ledger blocks.
"""

import pytest
from backend.ledger.block_schema import (
    BlockType,
    BlockSchemaContract,
    BlockValidationResult,
    validate_block_schema,
    get_default_contract,
)


class TestBlockSchemaContract:
    """Tests for BlockSchemaContract."""

    def test_default_required_fields(self):
        """Default contract should have required fields."""
        contract = BlockSchemaContract()
        assert "block_id" in contract.required_fields
        assert "prev_hash" in contract.required_fields
        assert "merkle_root" in contract.required_fields

    def test_version(self):
        """Contract should have version."""
        contract = BlockSchemaContract()
        assert contract.version == "1.0.0"


class TestValidateBlockSchema:
    """Tests for validate_block_schema function."""

    def test_valid_block(self):
        """Valid block should pass validation."""
        block = {
            "block_id": 1,
            "prev_hash": "abc123",
            "merkle_root": "def456",
            "timestamp": 1234567890,
            "proof_count": 10,
        }
        result = validate_block_schema(block)
        assert result.valid
        assert len(result.errors) == 0

    def test_missing_required_field(self):
        """Missing required field should fail validation."""
        block = {
            "block_id": 1,
            # Missing prev_hash, merkle_root, etc.
        }
        result = validate_block_schema(block)
        assert not result.valid
        assert len(result.errors) > 0
        assert any("prev_hash" in e for e in result.errors)

    def test_custom_contract(self):
        """Custom contract should be respected."""
        contract = BlockSchemaContract(required_fields=["custom_field"])
        block = {"other_field": "value"}
        result = validate_block_schema(block, contract)
        assert not result.valid
        assert any("custom_field" in e for e in result.errors)


class TestGetDefaultContract:
    """Tests for get_default_contract function."""

    def test_returns_contract(self):
        """Should return a BlockSchemaContract."""
        contract = get_default_contract()
        assert isinstance(contract, BlockSchemaContract)
