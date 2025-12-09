"""
Tests for PQ hash policy guard covering epoch governance rules.
"""

from substrate.crypto.pq_policy_guard import (
    BlockHeaderPQ,
    PQPolicyConfig,
    summarize_pq_policy_for_global_health,
    validate_block_pq_policy,
)


def _policy():
    return PQPolicyConfig.from_dict(
        {
            "epochs": [
                {
                    "name": "legacy",
                    "start_block": 0,
                    "end_block": 9,
                    "allowed_algorithms": ["sha256"],
                    "require_dual_commitment": False,
                    "require_legacy_hash": True,
                },
                {
                    "name": "transition",
                    "start_block": 10,
                    "end_block": 19,
                    "allowed_algorithms": ["sha256", "sha3-256"],
                    "require_dual_commitment": True,
                    "require_legacy_hash": True,
                },
                {
                    "name": "pq_only",
                    "start_block": 20,
                    "end_block": None,
                    "allowed_algorithms": ["sha3-256"],
                    "require_dual_commitment": True,
                    "require_legacy_hash": True,
                },
            ]
        }
    )


def test_legacy_epoch_without_pq_ok():
    policy = _policy()
    header = BlockHeaderPQ(
        block_number=5,
        algorithm_id="sha256",
        has_dual_commitment=False,
        has_legacy_hash=True,
    )

    verdict = validate_block_pq_policy(header, policy)

    assert verdict["ok"] is True
    assert verdict["epoch"] == "legacy"


def test_transition_epoch_missing_dual_commitment_blocks():
    policy = _policy()
    header = BlockHeaderPQ(
        block_number=12,
        algorithm_id="sha3-256",
        has_dual_commitment=False,
        has_legacy_hash=True,
    )

    verdict = validate_block_pq_policy(header, policy)

    assert verdict["ok"] is False
    assert verdict["code"] == "DUAL_COMMITMENT_REQUIRED"
    assert verdict["epoch"] == "transition"


def test_pq_only_epoch_missing_legacy_hash_blocks():
    policy = _policy()
    header = BlockHeaderPQ(
        block_number=25,
        algorithm_id="sha3-256",
        has_dual_commitment=True,
        has_legacy_hash=False,
    )

    verdict = validate_block_pq_policy(header, policy)

    assert verdict["ok"] is False
    assert verdict["code"] == "LEGACY_HASH_REQUIRED"
    assert verdict["epoch"] == "pq_only"


def test_summarize_pq_policy_for_global_health():
    policy = _policy()
    ok_verdict = validate_block_pq_policy(
        BlockHeaderPQ(5, "sha256", False, True),
        policy,
    )
    violation = validate_block_pq_policy(
        BlockHeaderPQ(15, "sha256", False, True),
        policy,
    )

    summary = summarize_pq_policy_for_global_health([ok_verdict, violation])

    assert summary["status"] == "alert"
    assert summary["violations"] == 1
    assert summary["total_checks"] == 2
    assert summary["current_epoch"] == violation["epoch"]
    assert summary["latest_block"] == violation["block_number"]
    assert summary["violation_codes"] == ["DUAL_COMMITMENT_REQUIRED"]
