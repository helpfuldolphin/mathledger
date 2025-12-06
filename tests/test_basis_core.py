import basis


def test_normalize_basic_implication():
    assert basis.normalize("p  ->  ( q -> r )") == "p->q->r"
    assert basis.normalize_pretty("p->q->r") == "p -> q -> r"
    assert basis.are_equivalent("p -> (q -> r)", "p -> q -> r")
    assert basis.atoms("(p /\\ q) -> r") == {"p", "q", "r"}


def test_merkle_root_and_proof():
    leaves = ["A", "B", "C"]
    root = basis.merkle_root(leaves)
    proof = basis.compute_merkle_proof(1, leaves)
    assert basis.verify_merkle_proof("B", proof, root)


def test_seal_block_deterministic():
    block = basis.seal_block(
        ["p -> q", "q -> r"],
        prev_hash="0" * 64,
        block_number=1,
        timestamp=1234.0,
    )
    assert block.header.block_number == 1
    assert block.statements == tuple(sorted(map(basis.normalize, ["p -> q", "q -> r"])))
    assert len(block.header.merkle_root) == 64
    assert block.header.prev_hash == "0" * 64


def test_dual_attestation_roundtrip():
    attestation = basis.build_attestation(
        reasoning_events=["a", "b"],
        ui_events=["x"],
        extra={"phase": "ix"},
    )
    assert basis.verify_attestation(attestation)
    assert attestation.reasoning_event_count == 2
    assert attestation.ui_event_count == 1
    assert len(attestation.composite_root) == 64
    # determinism
    attestation2 = basis.build_attestation(reasoning_events=["a", "b"], ui_events=["x"])
    assert attestation2.composite_root == attestation.composite_root


def test_curriculum_ladder_roundtrip(tmp_path):
    ladder = basis.CurriculumLadder(
        [
            basis.CurriculumTier(
                identifier="1",
                title="Foundations",
                description="Start here",
                objectives=("learn_arithmetic",),
            ),
            basis.CurriculumTier(
                identifier="2",
                title="Proofs",
                description="Deduction practice",
                prerequisites=("1",),
                objectives=("modus_ponens",),
            ),
        ]
    )
    path = tmp_path / "ladder.json"
    basis.ladder_to_json(ladder, path)
    loaded = basis.ladder_from_json(path)
    assert loaded.to_index()["2"].prerequisites == ("1",)
    assert len(list(loaded)) == 2

