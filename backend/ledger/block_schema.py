"""Block Schema Contract module.

Provides schema validation and contracts for ledger blocks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from enum import Enum


class BlockType(str, Enum):
    """Block type enumeration."""
    GENESIS = "genesis"
    STANDARD = "standard"
    CHECKPOINT = "checkpoint"


@dataclass
class BlockSchemaContract:
    """Contract defining required fields for a valid block."""
    required_fields: List[str] = field(default_factory=lambda: [
        "block_id",
        "prev_hash",
        "merkle_root",
        "timestamp",
        "proof_count",
    ])
    optional_fields: List[str] = field(default_factory=lambda: [
        "metadata",
        "attestation",
    ])
    version: str = "1.0.0"


@dataclass
class BlockValidationResult:
    """Result of block schema validation."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_block_schema(
    block: Dict[str, Any],
    contract: Optional[BlockSchemaContract] = None,
) -> BlockValidationResult:
    """Validate a block against the schema contract."""
    contract = contract or BlockSchemaContract()
    errors = []
    warnings = []

    for field_name in contract.required_fields:
        if field_name not in block:
            errors.append(f"Missing required field: {field_name}")

    return BlockValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def get_default_contract() -> BlockSchemaContract:
    """Get the default block schema contract."""
    return BlockSchemaContract()


__all__ = [
    "BlockType",
    "BlockSchemaContract",
    "BlockValidationResult",
    "validate_block_schema",
    "get_default_contract",
]
