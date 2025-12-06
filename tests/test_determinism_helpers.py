import string

from backend.repro.determinism import (
    deterministic_hash,
    deterministic_isoformat,
    deterministic_run_id,
    deterministic_seed_from_content,
    deterministic_slug,
)


def test_deterministic_hash_accepts_str_and_bytes():
    assert deterministic_hash("hello-world") == deterministic_hash(b"hello-world")


def test_deterministic_hash_normalizes_json_structures():
    payload_a = {"a": 1, "b": 2}
    payload_b = {"b": 2, "a": 1}
    assert deterministic_hash(payload_a) == deterministic_hash(payload_b)


def test_seed_and_timestamp_are_content_addressed():
    seed_a = deterministic_seed_from_content("proof", {"step": 1})
    seed_b = deterministic_seed_from_content("proof", {"step": 1})
    assert seed_a == seed_b

    ts_a = deterministic_isoformat("proof", {"step": 1})
    ts_b = deterministic_isoformat("proof", {"step": 1})
    assert ts_a == ts_b


def test_run_id_and_slug_are_ascii_and_stable():
    run_id = deterministic_run_id("test", {"key": "value"})
    slug = deterministic_slug({"key": "value"}, length=16)

    assert run_id.startswith("test-")
    assert slug.isascii()
    assert all(ch in string.hexdigits for ch in slug)

    run_id_repeat = deterministic_run_id("test", {"key": "value"})
    slug_repeat = deterministic_slug({"key": "value"}, length=16)

    assert run_id == run_id_repeat
    assert slug == slug_repeat

