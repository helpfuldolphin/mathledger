"""
Tests for governance validation (Lawkeeper).

Validates:
- Governance chain threading
- Root chain threading
- Dual-root integrity
- Determinism scores
"""

import json
import pytest
from pathlib import Path
from backend.governance.validator import (
    LawkeeperValidator,
    GovernanceEntry,
    DeclaredRoot
)
from backend.crypto.hashing import sha256_hex, DOMAIN_BLCK


@pytest.fixture
def valid_governance_chain(tmp_path):
    """Create a valid governance chain with proper threading."""
    # Chain: genesis → entry1 → entry2
    genesis_sig = sha256_hex("genesis", domain=b'\x00')
    entry1_sig = sha256_hex("entry1", domain=b'\x00')
    entry2_sig = sha256_hex("entry2", domain=b'\x00')

    chain = {
        "version": "1.0.0",
        "exported_at": "2025-11-04T00:00:00Z",
        "entry_count": 3,
        "entries": [
            {
                "signature": genesis_sig,
                "prev_signature": "",
                "timestamp": "2025-11-01T00:00:00Z",
                "status": "CLEAN",
                "determinism_score": 100,
                "version": "1.0.0",
                "replay_success": True
            },
            {
                "signature": entry1_sig,
                "prev_signature": genesis_sig,
                "timestamp": "2025-11-01T01:00:00Z",
                "status": "CLEAN",
                "determinism_score": 100,
                "version": "1.0.0",
                "replay_success": True
            },
            {
                "signature": entry2_sig,
                "prev_signature": entry1_sig,
                "timestamp": "2025-11-01T02:00:00Z",
                "status": "CLEAN",
                "determinism_score": 100,
                "version": "1.0.0",
                "replay_success": True
            }
        ]
    }

    gov_path = tmp_path / "governance_chain.json"
    with open(gov_path, "w") as f:
        json.dump(chain, f)

    return gov_path


@pytest.fixture
def invalid_governance_chain(tmp_path):
    """Create an invalid governance chain with broken threading."""
    chain = {
        "version": "1.0.0",
        "exported_at": "2025-11-04T00:00:00Z",
        "entry_count": 2,
        "entries": [
            {
                "signature": "abc123",
                "prev_signature": "",
                "timestamp": "2025-11-01T00:00:00Z",
                "status": "CLEAN",
                "determinism_score": 100,
                "version": "1.0.0",
                "replay_success": True
            },
            {
                "signature": "def456",
                "prev_signature": "wrong_hash",  # Broken chain
                "timestamp": "2025-11-01T01:00:00Z",
                "status": "CLEAN",
                "determinism_score": 100,
                "version": "1.0.0",
                "replay_success": True
            }
        ]
    }

    gov_path = tmp_path / "governance_chain_broken.json"
    with open(gov_path, "w") as f:
        json.dump(chain, f)

    return gov_path


@pytest.fixture
def valid_declared_roots(tmp_path):
    """Create valid declared roots with proper chaining."""
    # Create block chain
    block1_data = json.dumps({
        "block_number": 1,
        "root_hash": sha256_hex("root1"),
        "sealed_at": "2025-11-01T00:00:00Z"
    }, sort_keys=True)
    block1_hash = sha256_hex(block1_data, domain=DOMAIN_BLCK)

    block2_data = json.dumps({
        "block_number": 2,
        "root_hash": sha256_hex("root2"),
        "sealed_at": "2025-11-01T01:00:00Z"
    }, sort_keys=True)
    block2_hash = sha256_hex(block2_data, domain=DOMAIN_BLCK)

    roots = {
        "version": "1.0.0",
        "exported_at": "2025-11-04T00:00:00Z",
        "block_count": 2,
        "roots": [
            {
                "block_number": 1,
                "root_hash": sha256_hex("root1"),
                "prev_hash": "",
                "statement_count": 10,
                "sealed_at": "2025-11-01T00:00:00Z"
            },
            {
                "block_number": 2,
                "root_hash": sha256_hex("root2"),
                "prev_hash": block1_hash,
                "statement_count": 15,
                "sealed_at": "2025-11-01T01:00:00Z"
            }
        ]
    }

    roots_path = tmp_path / "declared_roots.json"
    with open(roots_path, "w") as f:
        json.dump(roots, f)

    return roots_path


def test_valid_governance_chain(valid_governance_chain, tmp_path):
    """Test validation of a valid governance chain."""
    empty_roots = tmp_path / "empty_roots.json"
    with open(empty_roots, "w") as f:
        json.dump({"version": "1.0.0", "block_count": 0, "roots": []}, f)

    validator = LawkeeperValidator(
        governance_path=valid_governance_chain,
        roots_path=empty_roots,
        verbose=False
    )

    entries = validator.load_governance_chain()
    assert len(entries) == 3

    result = validator.validate_governance_threading(entries)
    assert result is True

    result = validator.validate_determinism_scores(entries)
    assert result is True


def test_invalid_governance_chain(invalid_governance_chain, tmp_path):
    """Test validation of an invalid governance chain."""
    empty_roots = tmp_path / "empty_roots.json"
    with open(empty_roots, "w") as f:
        json.dump({"version": "1.0.0", "block_count": 0, "roots": []}, f)

    validator = LawkeeperValidator(
        governance_path=invalid_governance_chain,
        roots_path=empty_roots,
        verbose=False
    )

    entries = validator.load_governance_chain()
    assert len(entries) == 2

    result = validator.validate_governance_threading(entries)
    assert result is False
    assert len(validator.errors) > 0


def test_valid_declared_roots(tmp_path, valid_declared_roots):
    """Test validation of valid declared roots."""
    empty_gov = tmp_path / "empty_gov.json"
    with open(empty_gov, "w") as f:
        json.dump({"version": "1.0.0", "entry_count": 0, "entries": []}, f)

    validator = LawkeeperValidator(
        governance_path=empty_gov,
        roots_path=valid_declared_roots,
        verbose=False
    )

    roots = validator.load_declared_roots()
    assert len(roots) == 2

    result = validator.validate_root_threading(roots)
    assert result is True

    result = validator.validate_dual_roots(roots)
    assert result is True


def test_low_determinism_score(tmp_path):
    """Test that low determinism scores are rejected."""
    chain = {
        "version": "1.0.0",
        "exported_at": "2025-11-04T00:00:00Z",
        "entry_count": 1,
        "entries": [
            {
                "signature": sha256_hex("entry"),
                "prev_signature": "",
                "timestamp": "2025-11-01T00:00:00Z",
                "status": "CLEAN",
                "determinism_score": 80,  # Below threshold
                "version": "1.0.0",
                "replay_success": True
            }
        ]
    }

    gov_path = tmp_path / "low_score.json"
    with open(gov_path, "w") as f:
        json.dump(chain, f)

    empty_roots = tmp_path / "empty_roots.json"
    with open(empty_roots, "w") as f:
        json.dump({"version": "1.0.0", "block_count": 0, "roots": []}, f)

    validator = LawkeeperValidator(
        governance_path=gov_path,
        roots_path=empty_roots,
        verbose=False
    )

    entries = validator.load_governance_chain()
    result = validator.validate_determinism_scores(entries)
    assert result is False
    assert len(validator.errors) > 0


def test_full_adjudication(valid_governance_chain, valid_declared_roots):
    """Test full lawfulness adjudication."""
    validator = LawkeeperValidator(
        governance_path=valid_governance_chain,
        roots_path=valid_declared_roots,
        verbose=False
    )

    lawful = validator.adjudicate()
    assert lawful is True
    assert len(validator.errors) == 0


def test_missing_files(tmp_path):
    """Test validation with missing files."""
    nonexistent_gov = tmp_path / "nonexistent.json"
    nonexistent_roots = tmp_path / "nonexistent_roots.json"

    validator = LawkeeperValidator(
        governance_path=nonexistent_gov,
        roots_path=nonexistent_roots,
        verbose=False
    )

    lawful = validator.adjudicate()
    assert lawful is False
    assert len(validator.errors) > 0


def test_governance_entry_dataclass():
    """Test GovernanceEntry dataclass."""
    entry = GovernanceEntry(
        index=0,
        signature="abc123",
        prev_signature="",
        timestamp="2025-11-01T00:00:00Z",
        status="CLEAN",
        determinism_score=100
    )

    assert entry.index == 0
    assert entry.signature == "abc123"
    assert entry.determinism_score == 100


def test_declared_root_dataclass():
    """Test DeclaredRoot dataclass."""
    root = DeclaredRoot(
        block_number=1,
        root_hash="abc123",
        prev_hash="",
        statement_count=10,
        sealed_at="2025-11-01T00:00:00Z"
    )

    assert root.block_number == 1
    assert root.root_hash == "abc123"
    assert root.statement_count == 10
