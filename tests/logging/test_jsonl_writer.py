import json
from datetime import datetime

import pytest

from backend.logging.jsonl_writer import JsonlWriter


def test_write_appends_each_json_line(tmp_path):
    log_path = tmp_path / "harness.jsonl"
    writer = JsonlWriter(str(log_path))

    payloads = [
        {"run_id": 1, "status": "scheduled"},
        {"run_id": 2, "status": "complete"},
    ]
    for payload in payloads:
        writer.write(payload)
    writer.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == payloads


def test_close_prevents_further_writes(tmp_path):
    log_path = tmp_path / "closed.jsonl"
    writer = JsonlWriter(str(log_path))
    writer.close()

    with pytest.raises(ValueError):
        writer.write({"oops": True})


def test_context_manager_closes_writer(tmp_path):
    log_path = tmp_path / "ctx.jsonl"
    with JsonlWriter(str(log_path)) as writer:
        writer.write({"event": "start"})

    assert writer.closed
    assert json.loads(log_path.read_text(encoding="utf-8").strip()) == {"event": "start"}


def test_each_line_is_valid_json(tmp_path):
    log_path = tmp_path / "multi.jsonl"
    writer = JsonlWriter(str(log_path))
    for idx in range(5):
        writer.write({"idx": idx, "values": [idx, idx + 1]})
    writer.close()

    with log_path.open(encoding="utf-8") as log_file:
        for line in log_file:
            json.loads(line)


def test_harness_integration_preserves_order(tmp_path):
    """Simulate a harness run that emits deterministic cycle records."""
    log_path = tmp_path / "synthetic" / "first_light.jsonl"
    records = [
        {
            "cycle": idx,
            "delta_p": round(0.05 + idx * 0.01, 4),
            "timestamp": datetime(2025, 1, idx, 12, 0, tzinfo=None),
        }
        for idx in range(1, 6)
    ]

    with JsonlWriter(str(log_path), json_kwargs={"default": str}) as writer:
        for record in records:
            writer.write(record)

    parsed = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    expected = [
        {
            "cycle": rec["cycle"],
            "delta_p": rec["delta_p"],
            "timestamp": str(rec["timestamp"]),
        }
        for rec in records
    ]
    assert parsed == expected


def test_writer_preserves_key_order_and_compact_format(tmp_path):
    log_path = tmp_path / "ordered.jsonl"
    record = {"alpha": 1, "beta": 2, "label": "Δ"}

    writer = JsonlWriter(str(log_path))
    writer.write(record)
    writer.close()

    raw = log_path.read_text(encoding="utf-8").strip()
    assert raw == '{"alpha":1,"beta":2,"label":"Δ"}', "Writer must emit compact JSON with UTF-8 characters intact"


def test_writer_emits_literal_unicode_and_ascii_in_compact_form(tmp_path):
    """Writer must keep insertion order, compact separators, and literal Unicode."""
    log_path = tmp_path / "unicode.jsonl"
    record = {"alpha": "A", "pi": "π", "kanji": "漢字", "count": 1}

    writer = JsonlWriter(str(log_path))
    writer.write(record)
    writer.close()

    raw = log_path.read_text(encoding="utf-8").strip()
    assert raw == '{"alpha":"A","pi":"π","kanji":"漢字","count":1}'
